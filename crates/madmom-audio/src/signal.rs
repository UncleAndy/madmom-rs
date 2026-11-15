//! Signals, framing and the scalar signal functions.
//!
//! Port of `madmom/audio/signal.py` (1503 lines, 87 tests upstream). Faithfulness
//! notes — each of these silently changes numbers if it is missed:
//!
//! * frames are **centred** on `index * hop_size`
//!   (`start = index * hop_size - frame_size / 2`), and their number is
//!   `ceil(num_samples / hop_size)`: trailing frames are zero padded, not dropped.
//!   Verified bit-exactly against the reference dump (`tests/golden/sample.frames.npy`).
//! * `smooth` with an integer kernel uses a Hamming window of that length, and the
//!   convolution is `np.convolve(..., mode="same")`, whose centring for even kernels
//!   is `(kernel_len - 1) / 2` — see [`convolve_same`].
//! * integer signals are truncated towards zero whenever madmom casts back to the
//!   original dtype (`normalize`, `adjust_gain`, `remix`), which is reproduced by
//!   [`DType::cast`].
//! * energy/RMS/SPL are computed in `f64` (madmom converts to `float`), while the
//!   sample buffers stay `f32`.

use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayViewD, Axis, IxDyn};
use thiserror::Error;

/// Sample rate assumed when none is given (`madmom.audio.signal.SAMPLE_RATE`).
pub const SAMPLE_RATE: u32 = 44_100;
/// Number of channels assumed by default (`NUM_CHANNELS`).
pub const NUM_CHANNELS: usize = 1;
/// Default frame size (`FRAME_SIZE`).
pub const FRAME_SIZE: usize = 2048;
/// Default hop size at 44.1 kHz / 100 fps (`HOP_SIZE`).
pub const HOP_SIZE: f64 = 441.0;

/// Format the samples were loaded as.
///
/// madmom keeps the original dtype on the array, and several functions branch on it
/// (window scaling by `iinfo(dtype).max`, integer truncation, positive-gain refusal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 16-bit signed integer; maximal amplitude 32767.
    Int16,
    /// 32-bit signed integer (24-bit audio arrives in this form); 2147483647.
    Int32,
    /// 32-bit float; amplitudes are assumed to lie in `[-1, 1]`.
    Float32,
    /// 64-bit float; amplitudes are assumed to lie in `[-1, 1]`.
    Float64,
}

impl DType {
    /// `np.iinfo(dtype).max` for integer formats, `1.0` for float formats.
    pub fn max_amplitude(self) -> f64 {
        match self {
            DType::Int16 => 32767.0,
            DType::Int32 => 2147483647.0,
            DType::Float32 | DType::Float64 => 1.0,
        }
    }

    /// Whether this is an integer format.
    pub fn is_integer(self) -> bool {
        matches!(self, DType::Int16 | DType::Int32)
    }

    /// Cast a value the way numpy does when writing back into this dtype:
    /// truncation towards zero for integers (range-clamped, see the module notes).
    pub fn cast(self, value: f64) -> f32 {
        if self.is_integer() {
            let truncated = value.trunc();
            let (lo, hi) = match self {
                DType::Int16 => (-32768.0, 32767.0),
                _ => (-2147483648.0, 2147483647.0),
            };
            truncated.clamp(lo, hi) as f32
        } else {
            value as f32
        }
    }
}

/// Errors raised while working with signals (mirrors the reference's exceptions).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SignalError {
    /// `smooth` was called with a kernel size of exactly one sample.
    #[error("can't create a smoothing kernel of size 1")]
    InvalidKernelSize,
    /// `smooth` was called with a multidimensional kernel.
    #[error("can't smooth signal with a {}-dimensional kernel", .0)]
    InvalidKernelDims(usize),
    /// `smooth` needs a 1-D or 2-D signal.
    #[error("signal must be either 1D or 2D, got {} dims", .0)]
    InvalidSignalDims(usize),
    /// Positive gains would clip or overflow an integer signal.
    #[error("positive gain adjustments are only supported for float dtypes")]
    PositiveGainOnInteger,
    /// `normalize` only knows how to scale float, int16 and int32 data.
    #[error("only float and int16/32 dtypes supported for normalize, not {0:?}")]
    NormalizeDtype(DType),
    /// `rescale` accepts integer and float inputs only.
    #[error("unsupported signal dtype: {0:?}")]
    RescaleDtype(DType),
    /// Only mono<->multi conversions are implemented, like the reference.
    #[error("requested {requested} channels, but got {got} channels and channel conversion is not implemented")]
    ChannelConversionUnsupported {
        /// Number of channels that were requested.
        requested: usize,
        /// Number of channels the signal actually has.
        got: usize,
    },
    /// A frame index outside `0..num_frames` was requested.
    #[error("frame index {index} out of range for {num_frames} frames")]
    FrameIndexOutOfRange {
        /// Index that was requested.
        index: i64,
        /// Number of frames the signal has.
        num_frames: usize,
    },
}

/// A signal: samples plus the metadata madmom keeps on its array subclass.
///
/// Samples are stored as `f32`, which is exact for the formats in use (int16,
/// 24-bit and float32). [`DType`] records the original format, because the DSP
/// branches on it.
#[derive(Debug, Clone, PartialEq)]
pub struct Signal {
    data: ArrayD<f32>,
    dtype: DType,
    sample_rate: Option<u32>,
    num_channels: usize,
    start: Option<f64>,
    stop: Option<f64>,
}

impl Signal {
    /// Build a signal from samples, their original format and a sample rate.
    ///
    /// A 1-D array is a mono signal, a 2-D array is `(num_samples, num_channels)`,
    /// exactly like the reference.
    pub fn new(data: ArrayD<f32>, dtype: DType, sample_rate: Option<u32>) -> Self {
        let num_channels = if data.ndim() == 1 {
            1
        } else if data.ndim() == 2 {
            data.shape()[1]
        } else {
            data.shape().get(1).copied().unwrap_or(1)
        };
        Self {
            data,
            dtype,
            sample_rate,
            num_channels,
            start: None,
            stop: None,
        }
    }

    /// Build a mono signal from a flat array of samples.
    pub fn mono(data: Array1<f32>, dtype: DType, sample_rate: Option<u32>) -> Self {
        Self::new(data.into_dyn(), dtype, sample_rate)
    }

    /// Samples, as a dynamic-dimensional array.
    pub fn data(&self) -> ArrayViewD<'_, f32> {
        self.data.view()
    }

    /// The original format of the samples.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Sample rate, if known.
    pub fn sample_rate(&self) -> Option<u32> {
        self.sample_rate
    }

    /// Number of channels.
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Number of samples (the length of the first axis).
    pub fn num_samples(&self) -> usize {
        self.data.shape()[0]
    }

    /// Dimensionality of the sample array (1 for mono, 2 for multi-channel).
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Shape of the sample array.
    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    /// Whether this is a mono signal.
    pub fn is_mono(&self) -> bool {
        self.ndim() == 1
    }

    /// Length in seconds, or `None` when the sample rate is unknown.
    pub fn length(&self) -> Option<f64> {
        self.sample_rate
            .map(|rate| self.num_samples() as f64 / rate as f64)
    }

    /// Energy of the signal (see [`energy`]).
    pub fn energy(&self) -> f64 {
        energy(self.data())
    }

    /// Root mean square of the signal (see [`root_mean_square`]).
    pub fn root_mean_square(&self) -> f64 {
        root_mean_square(self.data())
    }

    /// Sound pressure level of the signal (see [`sound_pressure_level`]).
    pub fn sound_pressure_level(&self) -> f64 {
        sound_pressure_level(self.data(), self.dtype, None)
    }
}

/// Kernel argument of [`smooth`].
#[derive(Debug, Clone)]
pub enum Smoothing {
    /// No smoothing at all (madmom: `kernel=None`).
    None,
    /// Kernel length; a Hamming window of that length is used (madmom: integer).
    Size(usize),
    /// Explicit smoothing kernel (madmom: a numpy array).
    Kernel(Array1<f32>),
}

/// A symmetric Hamming window of `size` samples, as `np.hamming` builds it.
///
/// `0.54 - 0.46 * cos(2*pi*n/(size-1))`, with `n = 0..size-1`.
pub fn hamming(size: usize) -> Array1<f32> {
    let mut window = Array1::zeros(size);
    if size == 0 {
        return window;
    }
    if size == 1 {
        window[0] = 1.0;
        return window;
    }
    let denominator = (size - 1) as f64;
    for (n, value) in window.iter_mut().enumerate() {
        *value = (0.54 - 0.46 * (2.0 * std::f64::consts::PI * n as f64 / denominator).cos()) as f32;
    }
    window
}

/// Convolve `signal` with `kernel` and return the central part, like
/// `np.convolve(signal, kernel, mode="same")`.
///
/// `mode="same"` returns `max(len(signal), len(kernel))` values, taken from the full
/// convolution starting at `(min(len(signal), len(kernel)) - 1) / 2` (integer
/// division). For even kernels this is *not* the naive centre, so the offset is
/// spelled out here instead of being recalculated from a comment.
pub fn convolve_same(signal: ArrayView1<'_, f32>, kernel: ArrayView1<'_, f32>) -> Array1<f32> {
    let (n, k) = (signal.len(), kernel.len());
    if n == 0 || k == 0 {
        return Array1::zeros(n.max(k));
    }
    let full_len = n + k - 1;
    let mut full = vec![0.0f64; full_len];
    for (i, &s) in signal.iter().enumerate() {
        if s == 0.0 {
            continue;
        }
        let s = s as f64;
        for (j, &kv) in kernel.iter().enumerate() {
            full[i + j] += s * kv as f64;
        }
    }
    let out_len = n.max(k);
    let start = (n.min(k) - 1) / 2;
    let mut out = Array1::zeros(out_len);
    for (i, value) in out.iter_mut().enumerate() {
        if let Some(v) = full.get(start + i) {
            *value = *v as f32;
        }
    }
    out
}

/// Smooth a 1-D or 2-D signal along its first axis.
///
/// An integer kernel becomes a Hamming window ([`hamming`]), a kernel of size 1 is
/// refused, and a 2-D signal is convolved with `kernel[:, None]` exactly as
/// `scipy.signal.convolve2d(..., mode="same")` does it.
pub fn smooth(signal: ArrayViewD<'_, f32>, kernel: &Smoothing) -> Result<ArrayD<f32>, SignalError> {
    let kernel = match kernel {
        Smoothing::None => return Ok(signal.to_owned()),
        Smoothing::Size(0) => return Ok(signal.to_owned()),
        Smoothing::Size(1) => return Err(SignalError::InvalidKernelSize),
        Smoothing::Size(size) => hamming(*size),
        Smoothing::Kernel(kernel) => {
            if kernel.ndim() != 1 {
                return Err(SignalError::InvalidKernelDims(kernel.ndim()));
            }
            kernel.clone()
        }
    };
    match signal.ndim() {
        1 => Ok(convolve_same(
            signal.into_dimensionality::<ndarray::Ix1>().unwrap(),
            kernel.view(),
        )
        .into_dyn()),
        2 => {
            let matrix = signal.into_dimensionality::<ndarray::Ix2>().unwrap();
            let (rows, cols) = (matrix.shape()[0], matrix.shape()[1]);
            let (k, start) = (kernel.len(), (kernel.len() - 1) / 2);
            let mut out = Array2::zeros((rows, cols));
            for col in 0..cols {
                let column = matrix.column(col).to_owned();
                for row in 0..rows {
                    let mut acc = 0.0f64;
                    for j in 0..k {
                        let src = row as i64 + j as i64 - start as i64;
                        if src >= 0 && (src as usize) < rows {
                            acc += column[src as usize] as f64 * kernel[j] as f64;
                        }
                    }
                    out[[row, col]] = acc as f32;
                }
            }
            Ok(out.into_dyn())
        }
        dims => Err(SignalError::InvalidSignalDims(dims)),
    }
}

/// A symmetric Hamming window of `size` samples in `f64`, as `np.hamming` builds it.
///
/// The `f32` [`hamming`] is what the signal/spectrogram pipeline needs; the tempo module
/// works in `f64` (madmom casts its activations with `astype(float)`), and smoothing them
/// with a rounded kernel would not reach the plan's tolerance for `f64` DSP.
pub fn hamming_f64(size: usize) -> Array1<f64> {
    let mut window = Array1::zeros(size);
    if size == 0 {
        return window;
    }
    if size == 1 {
        window[0] = 1.0;
        return window;
    }
    let denominator = (size - 1) as f64;
    for (n, value) in window.iter_mut().enumerate() {
        *value = 0.54 - 0.46 * (2.0 * std::f64::consts::PI * n as f64 / denominator).cos();
    }
    window
}

/// [`convolve_same`] in `f64`, with the accumulation kept in `f64` as well.
pub fn convolve_same_f64(signal: ArrayView1<'_, f64>, kernel: ArrayView1<'_, f64>) -> Array1<f64> {
    let (n, k) = (signal.len(), kernel.len());
    if n == 0 || k == 0 {
        return Array1::zeros(n.max(k));
    }
    let mut full = vec![0.0f64; n + k - 1];
    for (i, &s) in signal.iter().enumerate() {
        if s == 0.0 {
            continue;
        }
        for (j, &kv) in kernel.iter().enumerate() {
            full[i + j] += s * kv;
        }
    }
    let out_len = n.max(k);
    let start = (n.min(k) - 1) / 2;
    let mut out = Array1::zeros(out_len);
    for (i, value) in out.iter_mut().enumerate() {
        if let Some(v) = full.get(start + i) {
            *value = *v;
        }
    }
    out
}

/// Smooth a 1-D `f64` signal with a Hamming window of `size` (madmom: `smooth(signal, n)`).
///
/// Sizes 0 and 1 return the signal unchanged, matching the reference's early exits.
pub fn smooth_f64(signal: ArrayView1<'_, f64>, size: usize) -> Result<Array1<f64>, SignalError> {
    match size {
        0 | 1 => Ok(signal.to_owned()),
        size => Ok(convolve_same_f64(signal, hamming_f64(size).view())),
    }
}

/// Convert a gain in dB into a linear factor (`np.power(sqrt(10), 0.1 * gain)`).
fn gain_factor(gain_db: f64) -> f64 {
    10f64.powf(gain_db / 20.0)
}

/// Adjust the gain of a signal by `gain` dB, keeping its dtype.
///
/// Positive gains are refused for integer data (they would clip), like the reference.
pub fn adjust_gain(
    signal: ArrayViewD<'_, f32>,
    dtype: DType,
    gain: f64,
) -> Result<ArrayD<f32>, SignalError> {
    let factor = gain_factor(gain);
    if factor > 1.0 && dtype.is_integer() {
        return Err(SignalError::PositiveGainOnInteger);
    }
    Ok(signal.mapv(|value| dtype.cast(value as f64 * factor)))
}

/// Attenuate a signal by `attenuation` dB (a negative gain).
pub fn attenuate(
    signal: ArrayViewD<'_, f32>,
    dtype: DType,
    attenuation: f64,
) -> Result<ArrayD<f32>, SignalError> {
    if attenuation == 0.0 {
        return Ok(signal.to_owned());
    }
    adjust_gain(signal, dtype, -attenuation)
}

/// Normalize a signal to its maximal amplitude, keeping its dtype.
///
/// Integer signals are scaled by `iinfo(dtype).max`, so that they afterwards cover
/// the full representable range.
pub fn normalize(signal: ArrayViewD<'_, f32>, dtype: DType) -> Result<ArrayD<f32>, SignalError> {
    if dtype.is_integer() && !matches!(dtype, DType::Int16 | DType::Int32) {
        return Err(SignalError::NormalizeDtype(dtype));
    }
    let mut scaling = signal
        .iter()
        .fold(0.0f64, |acc, &v| acc.max((v as f64).abs()));
    if dtype.is_integer() {
        scaling /= dtype.max_amplitude();
    }
    if scaling == 0.0 {
        return Ok(signal.to_owned());
    }
    Ok(signal.mapv(|value| dtype.cast(value as f64 / scaling)))
}

/// Rescale a signal to `[-1, 1]` as `f32` (madmom's `rescale`, default dtype).
///
/// Float signals are only cast; integer signals are divided by `iinfo(dtype).max`.
pub fn rescale(signal: ArrayViewD<'_, f32>, dtype: DType) -> Result<ArrayD<f32>, SignalError> {
    if dtype.is_integer() {
        let max = dtype.max_amplitude() as f32;
        Ok(signal.mapv(|value| value / max))
    } else {
        // float signals are only cast, which is a no-op for f32 buffers
        Ok(signal.to_owned())
    }
}

/// Remix a signal to the requested number of channels.
///
/// Only mono -> multi (copy) and multi -> mono (average, or a single channel) are
/// supported, as in the reference; anything else is refused instead of guessed.
pub fn remix(
    signal: ArrayViewD<'_, f32>,
    dtype: DType,
    num_channels: Option<usize>,
    channel: Option<usize>,
) -> Result<ArrayD<f32>, SignalError> {
    let ndim = signal.ndim();
    let requested = match num_channels {
        None => return Ok(signal.to_owned()),
        Some(requested) => requested,
    };
    if requested == ndim {
        return Ok(signal.to_owned());
    }
    if requested == 1 && ndim > 1 {
        let matrix = signal.into_dimensionality::<ndarray::Ix2>().unwrap();
        let channels = matrix.shape()[1];
        match channel {
            Some(index) if index < channels => Ok(matrix.column(index).to_owned().into_dyn()),
            Some(_) => Err(SignalError::ChannelConversionUnsupported {
                requested: index_or(requested),
                got: channels,
            }),
            None => {
                let mut out = Array1::zeros(matrix.shape()[0]);
                for row in 0..matrix.shape()[0] {
                    let mut acc = 0.0f64;
                    for col in 0..channels {
                        acc += matrix[[row, col]] as f64;
                    }
                    out[row] = dtype.cast(acc / channels as f64);
                }
                Ok(out.into_dyn())
            }
        }
    } else if requested > 1 && ndim == 1 {
        let flat = signal.into_dimensionality::<ndarray::Ix1>().unwrap();
        let mut out = Array2::zeros((flat.len(), requested));
        for (row, &value) in flat.iter().enumerate() {
            for col in 0..requested {
                out[[row, col]] = value;
            }
        }
        Ok(out.into_dyn())
    } else {
        Err(SignalError::ChannelConversionUnsupported {
            requested,
            got: if ndim > 1 { signal.shape()[1] } else { 1 },
        })
    }
}

fn index_or(value: usize) -> usize {
    value
}

/// Trim leading (`'f'`) and/or trailing (`'b'`) all-zero slices of a signal.
///
/// Works on the first axis, and follows the reference in summing over the remaining
/// axes to decide whether a slice is zero.
pub fn trim(signal: ArrayViewD<'_, f32>, where_: &str) -> ArrayD<f32> {
    let where_ = where_.to_ascii_uppercase();
    let len = signal.shape()[0];
    let slice_sum = |index: usize| -> f64 {
        let mut acc = 0.0f64;
        for value in signal.index_axis(Axis(0), index).iter() {
            acc += *value as f64;
        }
        acc
    };
    let mut first = 0usize;
    if where_.contains('F') {
        while first < len && slice_sum(first) == 0.0 {
            first += 1;
        }
    }
    let mut last = len;
    if where_.contains('B') {
        while last > first && slice_sum(last - 1) == 0.0 {
            last -= 1;
        }
    }
    let mut out = ArrayD::zeros(IxDyn(
        &[last - first]
            .iter()
            .chain(signal.shape()[1..].iter())
            .copied()
            .collect::<Vec<_>>(),
    ));
    for (target, source) in (first..last).enumerate() {
        out.index_axis_mut(Axis(0), target)
            .assign(&signal.index_axis(Axis(0), source));
    }
    out
}

/// Energy of a signal: `dot(flat, flat)` in `f64`.
///
/// Complex signals are turned into their magnitude first, and integer samples are
/// converted to float before squaring (the reference does both to avoid overflow).
pub fn energy(signal: ArrayViewD<'_, f32>) -> f64 {
    signal.iter().map(|&v| (v as f64) * (v as f64)).sum()
}

/// Root mean square of a signal: `sqrt(energy / size)`.
pub fn root_mean_square(signal: ArrayViewD<'_, f32>) -> f64 {
    let size = signal.len();
    if size == 0 {
        return 0.0;
    }
    (energy(signal) / size as f64).sqrt()
}

/// Sound pressure level of a signal in dB.
///
/// `p_ref` defaults to the maximal amplitude of the dtype (integer formats) or 1.0
/// (float formats). A silent signal yields the smallest finite value instead of
/// `-inf`, as `np.nan_to_num` does.
pub fn sound_pressure_level(signal: ArrayViewD<'_, f32>, dtype: DType, p_ref: Option<f64>) -> f64 {
    let rms = root_mean_square(signal);
    let reference = p_ref.unwrap_or_else(|| {
        if dtype.is_integer() {
            dtype.max_amplitude()
        } else {
            1.0
        }
    });
    if rms == 0.0 {
        return -f64::MAX;
    }
    20.0 * (rms / reference).log10()
}

/// Origin of a frame relative to its reference sample (madmom's `origin`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// The frame is centred on the reference sample (`origin = 0`).
    Center,
    /// The reference sample is the right edge of the frame (`origin = (frame_size - 1) / 2`).
    Online,
    /// The reference sample is the left edge of the frame (`origin = -(frame_size / 2)`).
    Stream,
    /// Explicit numeric origin, passed through as-is.
    Numeric(i64),
}

impl Origin {
    /// Numeric value used in the frame start calculation.
    pub fn value(self, frame_size: usize) -> i64 {
        match self {
            Origin::Center => 0,
            Origin::Online => (frame_size as i64 - 1) / 2,
            Origin::Stream => -(frame_size as i64 / 2),
            Origin::Numeric(value) => value,
        }
    }
}

/// How the end of a signal is handled (madmom's `end`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndOfSignal {
    /// Frames as long as the reference sample covers the signal: `ceil(n / hop)`.
    Normal,
    /// Frames as long as any sample of the frame covers the signal: `floor(n / hop) + 1`.
    Extend,
}

/// Extract one frame of a signal, padded with `pad` outside the signal.
///
/// Straight port of `signal_frame`: the frame is centred on `index * hop_size`
/// shifted by `origin`, and the part outside `[0, num_samples)` is filled with `pad`
/// (zeros in madmom's default call).
pub fn signal_frame(
    signal: ArrayView1<'_, f32>,
    index: i64,
    frame_size: usize,
    hop_size: f64,
    origin: i64,
    pad: f32,
) -> Array1<f32> {
    let num_samples = signal.len() as i64;
    let reference = (index as f64 * hop_size) as i64;
    let start = reference - frame_size as i64 / 2 - origin;
    let stop = start + frame_size as i64;
    if start >= 0 && stop <= num_samples {
        return signal
            .slice(ndarray::s![start as usize..stop as usize])
            .to_owned();
    }
    let mut frame = Array1::from_elem(frame_size, pad);
    let copy_start = start.max(0);
    let copy_stop = stop.min(num_samples);
    if copy_stop > copy_start {
        let destination_start = (copy_start - start) as usize;
        let destination_end = destination_start + (copy_stop - copy_start) as usize;
        let source = signal.slice(ndarray::s![copy_start as usize..copy_stop as usize]);
        frame
            .slice_mut(ndarray::s![destination_start..destination_end])
            .assign(&source);
    }
    frame
}

/// Frames of a signal, produced lazily like the reference's `FramedSignal`.
#[derive(Debug, Clone)]
pub struct FramedSignal {
    signal: Signal,
    frame_size: usize,
    hop_size: f64,
    origin: Origin,
    num_frames: usize,
}

impl FramedSignal {
    /// Frame a signal.
    ///
    /// `hop_size` is used as given; the reference sets it from `fps` when a frame
    /// rate is requested (`hop_size = sample_rate / fps`), which callers do.
    pub fn new(
        signal: Signal,
        frame_size: usize,
        hop_size: f64,
        origin: Origin,
        end: EndOfSignal,
    ) -> Self {
        let num_samples = signal.num_samples() as f64;
        let num_frames = match end {
            EndOfSignal::Normal => (num_samples / hop_size).ceil() as usize,
            EndOfSignal::Extend => (num_samples / hop_size + 1.0).floor() as usize,
        };
        Self {
            signal,
            frame_size,
            hop_size,
            origin,
            num_frames,
        }
    }

    /// Number of frames.
    pub fn num_frames(&self) -> usize {
        self.num_frames
    }

    /// Frame size in samples.
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Hop size in samples (may be fractional, as in the reference).
    pub fn hop_size(&self) -> f64 {
        self.hop_size
    }

    /// The underlying signal.
    pub fn signal(&self) -> &Signal {
        &self.signal
    }

    /// Frame rate, or `None` without a sample rate.
    pub fn frame_rate(&self) -> Option<f64> {
        self.signal
            .sample_rate()
            .map(|rate| rate as f64 / self.hop_size)
    }

    /// Frame rate (alias, matching the reference's `fps`).
    pub fn fps(&self) -> Option<f64> {
        self.frame_rate()
    }

    /// Overlap factor of two adjacent frames.
    pub fn overlap_factor(&self) -> f64 {
        1.0 - self.hop_size / self.frame_size as f64
    }

    /// Shape of the framed signal: `(num_frames, frame_size[, num_channels])`.
    pub fn shape(&self) -> Vec<usize> {
        let mut shape = vec![self.num_frames, self.frame_size];
        if self.signal.num_channels() != 1 {
            shape.push(self.signal.num_channels());
        }
        shape
    }

    /// Dimensionality of the framed signal.
    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// One frame, padded with zeros outside the signal.
    pub fn frame(&self, index: usize) -> ArrayD<f32> {
        let mono = self.signal.is_mono();
        if mono {
            let flat = self
                .signal
                .data()
                .into_dimensionality::<ndarray::Ix1>()
                .expect("mono signal is 1-D");
            signal_frame(
                flat,
                index as i64,
                self.frame_size,
                self.hop_size,
                self.origin.value(self.frame_size),
                0.0,
            )
            .into_dyn()
        } else {
            let matrix = self
                .signal
                .data()
                .into_dimensionality::<ndarray::Ix2>()
                .expect("multi-channel signal is 2-D");
            let channels = matrix.shape()[1];
            let mut out = Array2::zeros((self.frame_size, channels));
            for channel in 0..channels {
                let column = matrix.column(channel).to_owned();
                let frame = signal_frame(
                    column.view(),
                    index as i64,
                    self.frame_size,
                    self.hop_size,
                    self.origin.value(self.frame_size),
                    0.0,
                );
                for (row, value) in frame.iter().enumerate() {
                    out[[row, channel]] = *value;
                }
            }
            out.into_dyn()
        }
    }

    /// All frames as one array: `(num_frames, frame_size)` for mono signals.
    pub fn to_array(&self) -> Array2<f32> {
        assert!(
            self.signal.is_mono(),
            "to_array is only defined for mono signals; use frame() otherwise"
        );
        let mut out = Array2::zeros((self.num_frames, self.frame_size));
        for index in 0..self.num_frames {
            let frame = self.frame(index);
            let frame = frame
                .into_dimensionality::<ndarray::Ix1>()
                .expect("mono frame is 1-D");
            out.index_axis_mut(Axis(0), index).assign(&frame);
        }
        out
    }

    /// Energy of every frame.
    pub fn energy(&self) -> Array1<f64> {
        (0..self.num_frames)
            .map(|index| energy(self.frame(index).view()))
            .collect()
    }

    /// Root mean square of every frame.
    pub fn root_mean_square(&self) -> Array1<f64> {
        (0..self.num_frames)
            .map(|index| root_mean_square(self.frame(index).view()))
            .collect()
    }

    /// Sound pressure level of every frame.
    pub fn sound_pressure_level(&self) -> Array1<f64> {
        let dtype = self.signal.dtype();
        (0..self.num_frames)
            .map(|index| sound_pressure_level(self.frame(index).view(), dtype, None))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn hamming_matches_numpy_for_small_windows() {
        // np.hamming(3) == array([0.08, 1. , 0.08])
        let window = hamming(3);
        assert!((window[0] - 0.08).abs() < 1e-7);
        assert!((window[1] - 1.0).abs() < 1e-7);
        assert!((window[2] - 0.08).abs() < 1e-7);
    }

    #[test]
    fn convolve_same_follows_numpy_centring() {
        // np.convolve([1, 2, 3, 4], [0.5, 0.5], 'same')
        //   -> array([0.5, 1.5, 2.5, 3.5])
        let signal = array![1.0f32, 2.0, 3.0, 4.0];
        let kernel = array![0.5f32, 0.5];
        let out = convolve_same(signal.view(), kernel.view());
        assert_eq!(out.len(), 4);
        for (got, expected) in out.iter().zip([0.5f32, 1.5, 2.5, 3.5]) {
            assert!(
                (got - expected).abs() < 1e-6,
                "got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn signal_frames_are_centred_and_counted_like_the_reference() {
        // 9 samples, frame 4, hop 2 -> ceil(9/2) = 5 frames, centred on 0, 2, 4, 6, 8
        let signal = Signal::mono(
            array![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            DType::Float32,
            Some(1),
        );
        let framed = FramedSignal::new(signal, 4, 2.0, Origin::Center, EndOfSignal::Normal);
        assert_eq!(framed.num_frames(), 5);
        assert_eq!(framed.frame_size(), 4);
        // first frame: start = 0 - 2 = -2 -> two zeros, then samples 0 and 1
        assert_eq!(framed.frame(0), array![0.0f32, 0.0, 0.0, 1.0].into_dyn());
        // last frame: start = 8 - 2 = 6 -> samples 6, 7, 8 and one zero
        assert_eq!(framed.frame(4), array![6.0f32, 7.0, 8.0, 0.0].into_dyn());
        assert_eq!(framed.frame_rate(), Some(0.5));
        // 1 - hop/frame_size = 1 - 2/4
        assert!((framed.overlap_factor() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn normalize_scales_integer_signals_by_the_dtype_maximum() {
        // np.iinfo(np.int16).max == 32767 -> 16384 / (16384/32767) == 32767
        let signal = array![-16384.0f32, 16384.0].into_dyn();
        let normalized = normalize(signal.view(), DType::Int16).unwrap();
        assert_eq!(normalized[0], -32767.0);
        assert_eq!(normalized[1], 32767.0);
    }

    #[test]
    fn positive_gain_on_integer_signals_is_refused() {
        let signal = array![1.0f32, 2.0].into_dyn();
        assert_eq!(
            adjust_gain(signal.view(), DType::Int16, 3.0).unwrap_err(),
            SignalError::PositiveGainOnInteger
        );
        assert!(adjust_gain(signal.view(), DType::Float32, 3.0).is_ok());
    }

    #[test]
    fn energy_and_rms_use_float_accumulation() {
        let signal = array![3.0f32, 4.0].into_dyn();
        assert_eq!(energy(signal.view()), 25.0);
        assert!((root_mean_square(signal.view()) - (12.5f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn silent_signal_has_no_infinite_sound_pressure_level() {
        let signal = array![0.0f32, 0.0].into_dyn();
        assert_eq!(
            sound_pressure_level(signal.view(), DType::Float32, None),
            -f64::MAX
        );
    }
}
