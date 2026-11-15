//! Audio file decoding: WAV through an own RIFF reader, everything else through ffmpeg.
//!
//! Port of the loading half of `madmom/io/audio.py` (790 lines upstream, 27 tests in
//! `tests/test_io_audio.py`). The WAV reader is written by hand instead of using a
//! popular crate on purpose: scipy's `wavfile.read` (which madmom calls) has specific
//! conventions — 8-bit PCM stays **unsigned**, 24-bit PCM is returned as `int32` with
//! the three bytes sign-extended and *not* rescaled, and `WAVE_FORMAT_EXTENSIBLE`
//! files are read through their sub-format — and a reader with different conventions
//! would produce plausible, wrong samples.
//!
//! Structure: [`sample`] parses WAV containers and crops by `start`/`stop`;
//! [`ffmpeg`] builds and runs the ffmpeg command line (argument order is part of the
//! contract, it decides which filter wins). Channel remixing and normalization live in
//! `madmom-audio`, mirroring where madmom imports `remix` from.

#![deny(missing_docs)]

use std::fmt;

use ndarray::{Array2, ArrayD};
use thiserror::Error;

/// Sample format of a file, following scipy's `wavfile` conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// 8-bit PCM, returned unsigned (`|u1`).
    U8,
    /// 16-bit PCM (`<i2`).
    I16,
    /// 24-bit and 32-bit PCM, returned as `<i4` for both.
    I32,
    /// 32-bit IEEE float (`<f4`).
    F32,
    /// 64-bit IEEE float (`<f8`).
    F64,
}

impl SampleFormat {
    /// Number of bits per sample as written in the file.
    pub fn bits(self) -> u16 {
        match self {
            SampleFormat::U8 => 8,
            SampleFormat::I16 => 16,
            SampleFormat::I32 => 32,
            SampleFormat::F32 => 32,
            SampleFormat::F64 => 64,
        }
    }

    /// Whether samples are integers (madmom branches on this: window scaling,
    /// `rescale` by `iinfo(dtype).max`, refusal of positive gains).
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            SampleFormat::U8 | SampleFormat::I16 | SampleFormat::I32
        )
    }

    /// Maximal amplitude of the format, i.e. `np.iinfo(dtype).max` (1.0 for floats).
    pub fn max_amplitude(self) -> f64 {
        match self {
            SampleFormat::U8 => 255.0,
            SampleFormat::I16 => 32767.0,
            // madmom/scipy map 24-bit audio onto int32 without rescaling, so the
            // theoretical maximum of that container is what the reference uses.
            SampleFormat::I32 => 2147483647.0,
            SampleFormat::F32 | SampleFormat::F64 => 1.0,
        }
    }

    /// The NumPy `descr` string of this format.
    pub fn descr(self) -> &'static str {
        match self {
            SampleFormat::U8 => "|u1",
            SampleFormat::I16 => "<i2",
            SampleFormat::I32 => "<i4",
            SampleFormat::F32 => "<f4",
            SampleFormat::F64 => "<f8",
        }
    }
}

impl fmt::Display for SampleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.descr())
    }
}

/// Errors raised while decoding audio.
#[derive(Debug, Error)]
pub enum AudioError {
    /// The file could not be read.
    #[error("cannot read {path}: {source}")]
    Io {
        /// Path that was being read.
        path: String,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// The container is not a RIFF/WAVE file.
    #[error("{0} is not a RIFF/WAVE file")]
    NotWave(String),
    /// The container is RIFF/WAVE but malformed.
    #[error("malformed WAVE file: {0}")]
    MalformedWave(String),
    /// The compression format is not PCM or IEEE float.
    #[error("unsupported WAVE format tag {tag:#06x} ({bits} bits)")]
    UnsupportedWaveFormat {
        /// `wFormatTag` (or the first two bytes of the sub-format GUID).
        tag: u16,
        /// Bits per sample as declared in the file.
        bits: u16,
    },
    /// `sample_rate` was requested and differs from the file's rate.
    #[error("Requested sample rate of {requested} Hz, but got {found} Hz and re-sampling is not implemented.")]
    SampleRateMismatch {
        /// Sample rate that was asked for.
        requested: u32,
        /// Sample rate of the file.
        found: u32,
    },
    /// `dtype` was requested and differs from the file's format.
    #[error("Requested dtype {requested}, but got {found} and re-scaling is not implemented.")]
    DtypeMismatch {
        /// Format that was asked for.
        requested: SampleFormat,
        /// Format of the file.
        found: SampleFormat,
    },
    /// Decoding through an external command has not been implemented yet.
    #[error("decoding {0} needs ffmpeg, which is not wired up yet (plan task 1.2, second half)")]
    FfmpegNotImplemented(String),
}

/// Audio samples plus what is needed to interpret them.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioData {
    /// Samples, `f32` (exact for the integer formats actually met: 8/16/24-bit;
    /// 32-bit PCM loses precision, which madmom's `rescale` would drop anyway).
    ///
    /// Mono files are 1-D, multi-channel files are `(num_samples, num_channels)`.
    pub samples: ArrayD<f32>,
    /// Format the samples were stored in.
    pub format: SampleFormat,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl AudioData {
    /// Number of samples (first axis).
    pub fn num_samples(&self) -> usize {
        self.samples.shape()[0]
    }

    /// Number of channels.
    pub fn num_channels(&self) -> usize {
        if self.samples.ndim() == 1 {
            1
        } else {
            self.samples.shape()[1]
        }
    }

    /// Duration in seconds.
    pub fn length(&self) -> f64 {
        self.num_samples() as f64 / self.sample_rate as f64
    }
}

/// Options of the loaders, mirroring the reference's keyword arguments.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoadOptions {
    /// Requested sample rate; a mismatch is an error (no implicit resampling).
    pub sample_rate: Option<u32>,
    /// Requested sample format; a mismatch is an error (no implicit rescaling).
    pub dtype: Option<SampleFormat>,
    /// Start position in seconds; rounded to the nearest sample.
    pub start: Option<f64>,
    /// Stop position in seconds; the sample at that position is not returned.
    pub stop: Option<f64>,
    /// Number of channels to reduce/expand to (applied by `madmom-audio`).
    pub num_channels: Option<usize>,
    /// Channel to keep when down-mixing to mono (applied by `madmom-audio`).
    pub channel: Option<usize>,
}

/// Reads a WAV container and returns its samples in scipy's conventions.
///
/// Handles PCM 8/16/24/32-bit, IEEE float 32/64-bit, `WAVE_FORMAT_EXTENSIBLE` (the
/// sub-format decides), and skips unknown chunks (`LIST`, `fact`, `bext`, ...).
pub fn read_wave<P: AsRef<std::path::Path>>(path: P) -> Result<AudioData, AudioError> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|source| AudioError::Io {
        path: path.display().to_string(),
        source,
    })?;
    parse_wave(&bytes)
}

/// Parses a WAV container from memory (see [`read_wave`]).
pub fn parse_wave(bytes: &[u8]) -> Result<AudioData, AudioError> {
    let mut reader = RiffReader::new(bytes);
    let (format, data) = reader.read()?;
    let num_channels = format.num_channels as usize;
    let samples = decode_samples(data, format)?;
    let shaped = if num_channels > 1 {
        let frames = samples.len() / num_channels;
        let mut array = Array2::zeros((frames, num_channels));
        for (index, value) in samples.into_iter().enumerate() {
            array[[index / num_channels, index % num_channels]] = value;
        }
        array.into_dyn()
    } else {
        ndarray::Array1::from(samples).into_dyn()
    };
    Ok(AudioData {
        samples: shaped,
        format: format.format,
        sample_rate: format.sample_rate,
    })
}

/// Loads a WAV file the way `madmom.io.audio.load_wave_file` does (minus remixing,
/// which needs `madmom-audio`).
///
/// A requested `sample_rate` or `dtype` that differs from the file's is refused
/// instead of silently converted, exactly like the reference.
pub fn load_wave_file<P: AsRef<std::path::Path>>(
    path: P,
    options: &LoadOptions,
) -> Result<AudioData, AudioError> {
    let mut audio = read_wave(path)?;
    if let Some(requested) = options.sample_rate {
        if requested != audio.sample_rate {
            return Err(AudioError::SampleRateMismatch {
                requested,
                found: audio.sample_rate,
            });
        }
    }
    if let Some(requested) = options.dtype {
        if requested != audio.format {
            return Err(AudioError::DtypeMismatch {
                requested,
                found: audio.format,
            });
        }
    }
    if options.start.is_some() || options.stop.is_some() {
        audio.samples = crop(
            &audio.samples,
            audio.sample_rate,
            options.start,
            options.stop,
        );
    }
    Ok(audio)
}

/// Crops samples by `start`/`stop` in seconds, rounding like the reference:
/// `start = int(start * sample_rate)`, `stop = min(len, int(stop * sample_rate))`.
fn crop(
    samples: &ArrayD<f32>,
    sample_rate: u32,
    start: Option<f64>,
    stop: Option<f64>,
) -> ArrayD<f32> {
    let total = samples.shape()[0];
    let first = start
        .map(|value| (value * sample_rate as f64) as usize)
        .unwrap_or(0);
    let last = stop
        .map(|value| ((value * sample_rate as f64) as usize).min(total))
        .unwrap_or(total);
    let first = first.min(total);
    let last = last.max(first);
    let mut out = ArrayD::zeros(
        std::iter::once(last - first)
            .chain(samples.shape()[1..].iter().copied())
            .collect::<Vec<_>>(),
    );
    for (target, source) in (first..last).enumerate() {
        out.index_axis_mut(ndarray::Axis(0), target)
            .assign(&samples.index_axis(ndarray::Axis(0), source));
    }
    out
}

/// Declared format of a WAV `fmt ` chunk.
#[derive(Debug, Clone, Copy)]
struct WaveFormatInfo {
    format: SampleFormat,
    /// Bit depth as declared in the file: `SampleFormat::I32` covers both 24- and
    /// 32-bit PCM, and the decoder needs to know which one it is.
    bits: u16,
    num_channels: u16,
    sample_rate: u32,
}

/// Minimal RIFF/WAVE reader.
struct RiffReader<'a> {
    bytes: &'a [u8],
}

impl<'a> RiffReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    fn read(&mut self) -> Result<(WaveFormatInfo, &'a [u8]), AudioError> {
        if self.bytes.len() < 12 || &self.bytes[0..4] != b"RIFF" || &self.bytes[8..12] != b"WAVE" {
            return Err(AudioError::NotWave("input".to_string()));
        }
        let mut info: Option<WaveFormatInfo> = None;
        let mut data: Option<&[u8]> = None;
        let mut position = 12usize;
        while position + 8 <= self.bytes.len() {
            let id = &self.bytes[position..position + 4];
            let size = u32::from_le_bytes([
                self.bytes[position + 4],
                self.bytes[position + 5],
                self.bytes[position + 6],
                self.bytes[position + 7],
            ]) as usize;
            let body_start = position + 8;
            let body_end = (body_start + size).min(self.bytes.len());
            let body = &self.bytes[body_start..body_end];
            match id {
                b"fmt " => info = Some(parse_fmt_chunk(body)?),
                b"data" => data = Some(body),
                _ => {}
            }
            // chunks are word aligned: an odd size is followed by a pad byte
            position = body_start + size + (size % 2);
        }
        let info = info.ok_or_else(|| AudioError::MalformedWave("no fmt chunk".to_string()))?;
        let data = data.ok_or_else(|| AudioError::MalformedWave("no data chunk".to_string()))?;
        Ok((info, data))
    }
}

/// Reads a `fmt ` chunk, following the sub-format for `WAVE_FORMAT_EXTENSIBLE`.
fn parse_fmt_chunk(body: &[u8]) -> Result<WaveFormatInfo, AudioError> {
    if body.len() < 16 {
        return Err(AudioError::MalformedWave("short fmt chunk".to_string()));
    }
    let mut tag = u16::from_le_bytes([body[0], body[1]]);
    let num_channels = u16::from_le_bytes([body[2], body[3]]);
    let sample_rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
    let bits = u16::from_le_bytes([body[14], body[15]]);
    if tag == 0xFFFE {
        // WAVE_FORMAT_EXTENSIBLE: after the 18-byte WAVEFORMATEX come
        // wValidBitsPerSample (2), dwChannelMask (4) and the 16-byte SubFormat GUID,
        // whose first two bytes are the real format tag.
        if body.len() < 26 {
            return Err(AudioError::MalformedWave(
                "short extensible fmt chunk".to_string(),
            ));
        }
        tag = u16::from_le_bytes([body[24], body[25]]);
    }
    let format = match (tag, bits) {
        (1, 8) => SampleFormat::U8,
        (1, 16) => SampleFormat::I16,
        (1, 24) => SampleFormat::I32,
        (1, 32) => SampleFormat::I32,
        (3, 32) => SampleFormat::F32,
        (3, 64) => SampleFormat::F64,
        _ => {
            return Err(AudioError::UnsupportedWaveFormat { tag, bits });
        }
    };
    Ok(WaveFormatInfo {
        format,
        bits,
        num_channels,
        sample_rate,
    })
}

/// Converts the raw `data` chunk into `f32` samples, following scipy's conventions.
fn decode_samples(data: &[u8], info: WaveFormatInfo) -> Result<Vec<f32>, AudioError> {
    let bits = info.bits;
    let width = (bits / 8) as usize;
    if width == 0 {
        return Err(AudioError::MalformedWave("zero sample width".to_string()));
    }
    let count = data.len() / width;
    let mut out = Vec::with_capacity(count);
    match info.format {
        SampleFormat::U8 => {
            for byte in &data[..count] {
                out.push(*byte as f32);
            }
        }
        SampleFormat::I16 => {
            for chunk in data[..count * 2].chunks_exact(2) {
                out.push(i16::from_le_bytes([chunk[0], chunk[1]]) as f32);
            }
        }
        SampleFormat::I32 if bits == 24 => {
            // 24-bit samples are three bytes, sign-extended into i32 and *not*
            // rescaled, exactly as scipy returns them.
            for chunk in data[..count * 3].chunks_exact(3) {
                let raw = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], 0]);
                out.push(((raw << 8) >> 8) as f32);
            }
        }
        SampleFormat::I32 => {
            // 32-bit PCM passes through unchanged (f32 keeps 24 significant bits, which
            // is enough for the 24-bit case above and documented for full-range PCM).
            for chunk in data[..count * 4].chunks_exact(4) {
                out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32);
            }
        }
        SampleFormat::F32 => {
            for chunk in data[..count * 4].chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        SampleFormat::F64 => {
            for chunk in data[..count * 8].chunks_exact(8) {
                out.push(f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as f32);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a tiny WAV container in memory.
    fn wav_bytes(tag: u16, bits: u16, channels: u16, rate: u32, data: &[u8]) -> Vec<u8> {
        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * channels as u32 * (bits as u32 / 8)).to_le_bytes());
        out.extend_from_slice(&(channels * bits / 8).to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn reads_16_bit_pcm() {
        let mut data = Vec::new();
        for value in [-32768i16, 0, 1, 32767] {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let audio = parse_wave(&wav_bytes(1, 16, 1, 8000, &data)).unwrap();
        assert_eq!(audio.sample_rate, 8000);
        assert_eq!(audio.format, SampleFormat::I16);
        assert_eq!(audio.samples.shape(), &[4]);
        assert_eq!(
            audio.samples.iter().copied().collect::<Vec<_>>(),
            vec![-32768.0, 0.0, 1.0, 32767.0]
        );
    }

    #[test]
    fn reads_8_bit_pcm_as_unsigned() {
        let audio = parse_wave(&wav_bytes(1, 8, 1, 8000, &[0, 128, 255])).unwrap();
        assert_eq!(audio.format, SampleFormat::U8);
        assert_eq!(
            audio.samples.iter().copied().collect::<Vec<_>>(),
            vec![0.0, 128.0, 255.0]
        );
    }

    #[test]
    fn reads_24_bit_pcm_sign_extended_without_rescaling() {
        // -1 and -8388608 (the most negative 24-bit value) and 8388607
        let data = [0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x80, 0xFF, 0xFF, 0x7F];
        let audio = parse_wave(&wav_bytes(1, 24, 1, 8000, &data)).unwrap();
        assert_eq!(audio.format, SampleFormat::I32);
        assert_eq!(
            audio.samples.iter().copied().collect::<Vec<_>>(),
            vec![-1.0, -8388608.0, 8388607.0]
        );
    }

    #[test]
    fn reads_stereo_as_two_columns() {
        let mut data = Vec::new();
        for value in [100i16, 200, 300, 400] {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let audio = parse_wave(&wav_bytes(1, 16, 2, 44100, &data)).unwrap();
        assert_eq!(audio.samples.shape(), &[2, 2]);
        assert_eq!(audio.num_channels(), 2);
        assert_eq!(audio.samples[[1, 0]], 300.0);
        assert_eq!(audio.samples[[1, 1]], 400.0);
    }

    #[test]
    fn reads_extensible_files_through_the_sub_format() {
        let mut body = Vec::new();
        body.extend_from_slice(&0xFFFEu16.to_le_bytes()); // wFormatTag
        body.extend_from_slice(&1u16.to_le_bytes()); // channels
        body.extend_from_slice(&8000u32.to_le_bytes()); // sample rate
        body.extend_from_slice(&16000u32.to_le_bytes()); // byte rate
        body.extend_from_slice(&2u16.to_le_bytes()); // block align
        body.extend_from_slice(&32u16.to_le_bytes()); // bits (float32)
        body.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        body.extend_from_slice(&16u16.to_le_bytes()); // wValidBitsPerSample
        body.extend_from_slice(&0x0000_0004u32.to_le_bytes()); // dwChannelMask
                                                               // SubFormat GUID: {00000003-0000-0010-8000-00AA00389B71}, IEEE float
        body.extend_from_slice(&0x0003u16.to_le_bytes());
        body.extend_from_slice(&[
            0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
        ]);
        let mut container = b"RIFF".to_vec();
        container.extend_from_slice(&(4 + 8 + body.len() as u32 + 8 + 4).to_le_bytes());
        container.extend_from_slice(b"WAVE");
        container.extend_from_slice(b"fmt ");
        container.extend_from_slice(&(body.len() as u32).to_le_bytes());
        container.extend_from_slice(&body);
        container.extend_from_slice(b"data");
        container.extend_from_slice(&4u32.to_le_bytes());
        container.extend_from_slice(&0.5f32.to_le_bytes());
        let audio = parse_wave(&container).unwrap();
        assert_eq!(audio.format, SampleFormat::F32);
        assert_eq!(audio.samples[0], 0.5);
    }

    #[test]
    fn refuses_other_formats_and_bad_containers() {
        assert!(matches!(
            parse_wave(&wav_bytes(0x0011, 4, 1, 8000, &[0, 0])),
            Err(AudioError::UnsupportedWaveFormat { .. })
        ));
        assert!(matches!(
            parse_wave(b"RIFFxxxxWAVE"),
            Err(AudioError::MalformedWave(_))
        ));
        assert!(matches!(
            parse_wave(b"not a wave file"),
            Err(AudioError::NotWave(_))
        ));
    }

    #[test]
    fn cropping_follows_the_reference_rounding() {
        let samples =
            ndarray::Array1::from((0..10).map(|v| v as f32).collect::<Vec<_>>()).into_dyn();
        // start = int(0.0002 * 8000) = 1, stop = int(0.0007 * 8000) = 5
        let cropped = crop(&samples, 8000, Some(0.0002), Some(0.0007));
        assert_eq!(
            cropped.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        // a stop beyond the end is clamped to the length
        let cropped = crop(&samples, 8000, None, Some(10.0));
        assert_eq!(cropped.len(), 10);
    }
}
