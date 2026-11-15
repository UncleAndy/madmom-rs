//! Spectrograms and their differences.
//!
//! Port of `madmom/audio/spectrogram.py` (1486 lines, 49 tests in
//! `tests/test_audio_spectrogram.py`). Details that decide the numbers:
//!
//! * a magnitude spectrogram is `|STFT|` and stays `float32`;
//! * filtering is a matrix product (`spectrogram · filterbank`) accumulating in
//!   `float32`, like the reference's `np.dot` on float32 operands;
//! * the logarithmic stage is `log(mul * x + add)` **in place** with `LOG = np.log10`,
//!   `MUL = ADD = 1`, so it is `log10(1 + x)` and not a natural logarithm;
//! * the difference stage prepends `diff_frames` rows of `+inf` to the spectrogram,
//!   differences those, and then sets every `inf` to `0` — that is how the first
//!   `diff_frames` differences become zero (the reference zeroes *any* infinity, a
//!   quirk that is reproduced here);
//! * `diff_frames` is derived from `diff_ratio` through the window shape:
//!   `argmax(window > ratio * max(window))`, `(len/2 - that) / hop`, rounded, at least 1
//!   (1 for ratio 0.5 and the standard 2048/441 setup, 2 for ratio 1.0);
//! * `stack_diffs` stacks the *spectrogram* with the differences (`hstack` in frequency
//!   direction), which is where the 90-wide input of the beat networks comes from.
//!
//! Not ported yet (plan task 1.6): `diff_max_bins > 1`, which needs scipy's
//! `maximum_filter` semantics, and the multi-band / semitone-bandpass variants.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex32;
use thiserror::Error;

use crate::filters::{
    bark_filterbank, logarithmic_filterbank, mel_filterbank, BarkSpacing, FilterError, Filterbank,
    A4, FMAX, FMIN, NUM_BANDS,
};
use crate::stft::{ShortTimeFourierTransform, WindowFunction};

/// Default ratio used to derive the number of difference frames.
pub const DIFF_RATIO: f64 = 0.5;
/// Default flag: keep negative differences as they are.
pub const POSITIVE_DIFFS: bool = false;
/// Default logarithmic scaling of the log spectrogram (`np.log10`).
pub const LOG_BASE: LogKind = LogKind::Log10;
/// Default multiplier of the logarithmic spectrogram.
pub const MUL: f32 = 1.0;
/// Default additive constant of the logarithmic spectrogram.
pub const ADD: f32 = 1.0;

/// Errors of the spectrogram stages.
#[derive(Debug, Error)]
pub enum SpectrogramError {
    /// A filterbank could not be built.
    #[error("cannot build the filterbank: {0}")]
    Filterbank(#[from] FilterError),
    /// A feature is documented as not ported yet.
    #[error("{0}")]
    NotImplemented(&'static str),
    /// The spectrogram has no bin frequencies, so a filterbank cannot be built for it.
    #[error("the spectrogram carries no bin frequencies")]
    MissingBinFrequencies,
    /// Stacking needs the spectrogram the difference was computed from.
    #[error("stacking the differences needs the underlying spectrogram")]
    MissingUnderlyingSpectrogram,
}

/// Which logarithm the logarithmic spectrogram uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogKind {
    /// `np.log10`, the reference's default.
    Log10,
    /// `np.log`.
    Ln,
}

impl LogKind {
    /// Apply the logarithm to one value.
    pub fn apply(self, value: f32) -> f32 {
        match self {
            LogKind::Log10 => value.log10(),
            LogKind::Ln => value.ln(),
        }
    }
}

/// How the differences are stacked with the spectrogram they were computed from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackDiffs {
    /// Return only the differences (`stack_diffs=None`).
    None,
    /// `np.hstack`: side by side in frequency direction.
    Horizontal,
    /// `np.vstack`: one below the other in time direction.
    Vertical,
    /// `np.dstack`: as a third (depth) dimension.
    Depth,
}

/// Magnitudes of a complex transform, `np.abs(stft)`.
pub fn spec(transform: ArrayView2<'_, Complex32>) -> Array2<f32> {
    transform.mapv(|value| value.norm())
}

/// Number of frames to difference over, derived from `diff_ratio` and the window.
///
/// Straight port of `_diff_frames`: the first window sample whose magnitude exceeds
/// `diff_ratio * max(window)` sets the effective window length, the difference to half
/// the window is converted to frames and rounded, with a minimum of one frame.
pub fn diff_frames(
    diff_ratio: f64,
    hop_size: f64,
    frame_size: usize,
    window: WindowFunction,
) -> usize {
    let window = window.build(frame_size);
    let maximum = window.iter().fold(0.0f64, |acc, value| acc.max(*value));
    let threshold = diff_ratio * maximum;
    let sample = window
        .iter()
        .position(|value| *value > threshold)
        .unwrap_or(0) as f64;
    let diff_samples = window.len() as f64 / 2.0 - sample;
    let frames = (diff_samples / hop_size).round();
    (frames as i64).max(1) as usize
}

/// Difference parameters, mirroring the reference's keyword arguments.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferenceParameters {
    /// Ratio of the window height at which the difference is taken.
    pub diff_ratio: f64,
    /// Explicit number of frames to difference over (overrides `diff_ratio`).
    pub diff_frames: Option<usize>,
    /// Width of a maximum filter applied in frequency direction (not ported yet).
    pub diff_max_bins: Option<usize>,
    /// Keep only positive differences.
    pub positive_diffs: bool,
    /// How to stack the differences with the spectrogram.
    pub stack_diffs: StackDiffs,
}

impl Default for DifferenceParameters {
    fn default() -> Self {
        Self {
            diff_ratio: DIFF_RATIO,
            diff_frames: None,
            diff_max_bins: None,
            positive_diffs: POSITIVE_DIFFS,
            stack_diffs: StackDiffs::None,
        }
    }
}

/// A spectrogram: magnitudes plus the metadata the later stages need.
///
/// The reference models the stages as subclasses of `Spectrogram`, each keeping a
/// reference to the one it was built from (`filterbank`, `spectrogram`, `stft`). This
/// struct keeps the same information flat, because the chain is used linearly anyway.
#[derive(Debug, Clone)]
pub struct Spectrogram {
    data: Array2<f32>,
    bin_frequencies: Option<Array1<f64>>,
    fps: Option<f64>,
    hop_size: f64,
    frame_size: usize,
    window: WindowFunction,
    filterbank: Option<Filterbank>,
    /// The spectrogram the difference was computed from (needed for stacking).
    underlying: Option<Box<Spectrogram>>,
    difference: Option<DifferenceParameters>,
}

impl Spectrogram {
    /// Magnitude spectrogram of a transform.
    pub fn from_transform(transform: &ShortTimeFourierTransform) -> Self {
        let frames = transform.frames();
        Self {
            data: spec(transform.values()),
            bin_frequencies: Some(transform.bin_frequencies()),
            fps: frames.fps(),
            hop_size: frames.hop_size(),
            frame_size: frames.frame_size(),
            window: WindowFunction::Hanning,
            filterbank: None,
            underlying: None,
            difference: None,
        }
    }

    /// Magnitudes of an existing spectrogram (used by the difference stage).
    pub fn from_values(data: Array2<f32>, bin_frequencies: Option<Array1<f64>>) -> Self {
        Self {
            data,
            bin_frequencies,
            fps: None,
            hop_size: 441.0,
            frame_size: 2048,
            window: WindowFunction::Hanning,
            filterbank: None,
            underlying: None,
            difference: None,
        }
    }

    /// The magnitudes, `(num_frames, num_bins)`.
    pub fn data(&self) -> ArrayView2<'_, f32> {
        self.data.view()
    }

    /// Number of frames.
    pub fn num_frames(&self) -> usize {
        self.data.shape()[0]
    }

    /// Number of bins (or bands).
    pub fn num_bins(&self) -> usize {
        self.data.shape()[1]
    }

    /// Frequencies of the bins, if known.
    pub fn bin_frequencies(&self) -> Option<ArrayView1<'_, f64>> {
        self.bin_frequencies.as_ref().map(|bins| bins.view())
    }

    /// Frame rate, if known.
    pub fn fps(&self) -> Option<f64> {
        self.fps
    }

    /// Hop size in samples.
    pub fn hop_size(&self) -> f64 {
        self.hop_size
    }

    /// Frame size in samples.
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// The filterbank this spectrogram was filtered with, if any.
    pub fn filterbank(&self) -> Option<&Filterbank> {
        self.filterbank.as_ref()
    }

    /// The spectrogram this one was computed from (the reference's `.spectrogram`).
    pub fn underlying(&self) -> Option<&Spectrogram> {
        self.underlying.as_deref()
    }

    /// The difference parameters, if this spectrogram is a difference.
    pub fn difference_parameters(&self) -> Option<&DifferenceParameters> {
        self.difference.as_ref()
    }

    /// Build the filterbank described by `settings` and filter with it.
    pub fn filter(&self, settings: &FilterbankSettings) -> Result<Self, SpectrogramError> {
        let bins = self
            .bin_frequencies
            .as_ref()
            .ok_or(SpectrogramError::MissingBinFrequencies)?;
        let filterbank = match settings.kind {
            FilterbankKind::Logarithmic => logarithmic_filterbank(
                bins.view(),
                settings.num_bands,
                settings.fmin,
                settings.fmax,
                settings.fref,
                settings.norm_filters,
                settings.unique_filters,
            )?,
            FilterbankKind::Mel => mel_filterbank(
                bins.view(),
                settings.num_bands,
                settings.fmin,
                settings.fmax,
                settings.norm_filters,
                settings.unique_filters,
            )?,
            FilterbankKind::Bark(spacing) => bark_filterbank(
                bins.view(),
                spacing,
                settings.fmin,
                settings.fmax,
                settings.norm_filters,
                settings.unique_filters,
            )?,
        };
        self.filter_with(filterbank)
    }

    /// Filter with an existing filterbank: `spectrogram · filterbank`.
    pub fn filter_with(&self, filterbank: Filterbank) -> Result<Self, SpectrogramError> {
        let mut out = Array2::<f32>::zeros((self.num_frames(), filterbank.num_bands()));
        for frame in 0..self.num_frames() {
            let spectrum = self.data.row(frame);
            for band in 0..filterbank.num_bands() {
                let mut acc = 0.0f32;
                for bin in 0..self.num_bins() {
                    if bin >= filterbank.num_bins() {
                        break;
                    }
                    acc += spectrum[bin] * filterbank.matrix()[[bin, band]];
                }
                out[[frame, band]] = acc;
            }
        }
        Ok(Self {
            data: out,
            bin_frequencies: Some(filterbank.center_frequencies()),
            fps: self.fps,
            hop_size: self.hop_size,
            frame_size: self.frame_size,
            window: self.window,
            filterbank: Some(filterbank),
            underlying: None,
            difference: None,
        })
    }

    /// Logarithmic scaling: `log(mul * x + add)`, in place like the reference.
    pub fn logarithmic(&self, log: Option<LogKind>, mul: Option<f32>, add: Option<f32>) -> Self {
        let mut data = self.data.clone();
        if let Some(mul) = mul {
            data.mapv_inplace(|value| value * mul);
        }
        if let Some(add) = add {
            data.mapv_inplace(|value| value + add);
        }
        if let Some(log) = log {
            data.mapv_inplace(|value| log.apply(value));
        }
        Self {
            data,
            bin_frequencies: self.bin_frequencies.clone(),
            fps: self.fps,
            hop_size: self.hop_size,
            frame_size: self.frame_size,
            window: self.window,
            filterbank: self.filterbank.clone(),
            underlying: None,
            difference: None,
        }
    }

    /// Temporal difference of the spectrogram.
    ///
    /// The first `diff_frames` rows of the result are zero, exactly as in the reference:
    /// `+inf` is prepended, differenced, and every infinity is then set to zero.
    /// With `stack_diffs` the underlying spectrogram is stacked to the differences.
    pub fn difference(&self, parameters: &DifferenceParameters) -> Result<Self, SpectrogramError> {
        if parameters.diff_max_bins.unwrap_or(1) > 1 {
            return Err(SpectrogramError::NotImplemented(
                "diff_max_bins > 1 needs scipy's maximum_filter semantics (plan task 1.6)",
            ));
        }
        let frames = match parameters.diff_frames {
            Some(frames) => frames,
            None => diff_frames(
                parameters.diff_ratio,
                self.hop_size,
                self.frame_size,
                self.window,
            ),
        };
        let (rows, bins) = (self.num_frames(), self.num_bins());
        // prepend diff_frames rows of +inf, then difference
        let mut padded = Array2::<f32>::from_elem((rows + frames, bins), f32::INFINITY);
        for row in 0..rows {
            padded
                .index_axis_mut(ndarray::Axis(0), row + frames)
                .assign(&self.data.index_axis(ndarray::Axis(0), row));
        }
        let mut diff = Array2::<f32>::zeros((rows, bins));
        for row in 0..rows {
            for bin in 0..bins {
                diff[[row, bin]] = padded[[row + frames, bin]] - padded[[row, bin]];
            }
        }
        if parameters.positive_diffs {
            diff.mapv_inplace(|value| value.max(0.0));
        }
        // the reference sets every infinity to zero (the prepended rows are gone by now,
        // but an infinity in the input would show up here as NaN/inf as well)
        diff.mapv_inplace(|value| if value.is_infinite() { 0.0 } else { value });

        let stacked = match parameters.stack_diffs {
            StackDiffs::None => diff,
            StackDiffs::Horizontal => {
                let mut out = Array2::<f32>::zeros((rows, bins * 2));
                for row in 0..rows {
                    for bin in 0..bins {
                        out[[row, bin]] = self.data[[row, bin]];
                        out[[row, bins + bin]] = diff[[row, bin]];
                    }
                }
                out
            }
            StackDiffs::Vertical => {
                let mut out = Array2::<f32>::zeros((rows * 2, bins));
                for row in 0..rows {
                    for bin in 0..bins {
                        out[[row, bin]] = self.data[[row, bin]];
                        out[[rows + row, bin]] = diff[[row, bin]];
                    }
                }
                out
            }
            StackDiffs::Depth => {
                return Err(SpectrogramError::NotImplemented(
                    "depth stacking needs a 3-D representation; use horizontal or vertical",
                ))
            }
        };
        Ok(Self {
            data: stacked,
            bin_frequencies: self.bin_frequencies.clone(),
            fps: self.fps,
            hop_size: self.hop_size,
            frame_size: self.frame_size,
            window: self.window,
            filterbank: self.filterbank.clone(),
            underlying: Some(Box::new(self.clone())),
            difference: Some(DifferenceParameters {
                diff_frames: Some(frames),
                ..parameters.clone()
            }),
        })
    }

    /// Positive differences only (`np.maximum(self, 0)`).
    pub fn positive_diff(&self) -> Self {
        let mut out = self.clone();
        out.data.mapv_inplace(|value| value.max(0.0));
        out
    }
}

/// Which filterbank a processor builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterbankKind {
    /// `LogarithmicFilterbank`, the default of the reference.
    Logarithmic,
    /// `MelFilterbank`.
    Mel,
    /// `BarkFilterbank`, with its spacing.
    Bark(BarkSpacing),
}

/// Processor turning a transform into a magnitude spectrogram.
#[derive(Debug, Clone, Default)]
pub struct SpectrogramProcessor;

impl SpectrogramProcessor {
    /// Compute the magnitudes.
    pub fn process(&self, transform: &ShortTimeFourierTransform) -> Spectrogram {
        Spectrogram::from_transform(transform)
    }
}

/// Settings of a filterbank, i.e. the reference's keyword arguments in one place.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterbankSettings {
    /// Filterbank to build.
    pub kind: FilterbankKind,
    /// Number of bands (bands per octave for the logarithmic filterbank).
    pub num_bands: usize,
    /// Minimum frequency.
    pub fmin: f64,
    /// Maximum frequency.
    pub fmax: f64,
    /// Tuning frequency.
    pub fref: f64,
    /// Normalize the filters.
    pub norm_filters: bool,
    /// Remove duplicate bins.
    pub unique_filters: bool,
}

impl Default for FilterbankSettings {
    fn default() -> Self {
        Self {
            kind: FilterbankKind::Logarithmic,
            num_bands: NUM_BANDS,
            fmin: FMIN,
            fmax: FMAX,
            fref: A4,
            norm_filters: true,
            unique_filters: true,
        }
    }
}

/// Processor turning a spectrogram into a filtered spectrogram.
#[derive(Debug, Clone, Default)]
pub struct FilteredSpectrogramProcessor {
    /// Filterbank settings.
    pub settings: FilterbankSettings,
}

impl FilteredSpectrogramProcessor {
    /// Filter a spectrogram.
    pub fn process(&self, spectrogram: &Spectrogram) -> Result<Spectrogram, SpectrogramError> {
        spectrogram.filter(&self.settings)
    }
}

/// Processor applying a logarithm to a spectrogram.
#[derive(Debug, Clone)]
pub struct LogarithmicSpectrogramProcessor {
    /// Logarithm to apply, `None` to skip.
    pub log: Option<LogKind>,
    /// Multiplier applied before the logarithm.
    pub mul: Option<f32>,
    /// Additive constant applied before the logarithm.
    pub add: Option<f32>,
}

impl Default for LogarithmicSpectrogramProcessor {
    fn default() -> Self {
        Self {
            log: Some(LOG_BASE),
            mul: Some(MUL),
            add: Some(ADD),
        }
    }
}

impl LogarithmicSpectrogramProcessor {
    /// Apply the scaling.
    pub fn process(&self, spectrogram: &Spectrogram) -> Spectrogram {
        spectrogram.logarithmic(self.log, self.mul, self.add)
    }
}

/// Processor filtering *and* scaling in one step (the reference's
/// `LogarithmicFilteredSpectrogramProcessor`, which the beat chains use).
#[derive(Debug, Clone, Default)]
pub struct LogarithmicFilteredSpectrogramProcessor {
    /// Filterbank settings.
    pub filter: FilteredSpectrogramProcessor,
    /// Logarithmic settings.
    pub log: LogarithmicSpectrogramProcessor,
}

impl LogarithmicFilteredSpectrogramProcessor {
    /// Filter, then apply the logarithm.
    pub fn process(&self, spectrogram: &Spectrogram) -> Result<Spectrogram, SpectrogramError> {
        let filtered = self.filter.process(spectrogram)?;
        Ok(self.log.process(&filtered))
    }
}

/// Processor computing the temporal difference of a spectrogram.
#[derive(Debug, Clone, Default)]
pub struct SpectrogramDifferenceProcessor {
    /// Difference parameters.
    pub parameters: DifferenceParameters,
}

impl SpectrogramDifferenceProcessor {
    /// Compute the difference (and stack it, if requested).
    pub fn process(&self, spectrogram: &Spectrogram) -> Result<Spectrogram, SpectrogramError> {
        spectrogram.difference(&self.parameters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn spec_is_the_magnitude() {
        let transform = array![
            [Complex32::new(3.0, 4.0), Complex32::new(0.0, -1.0)],
            [Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)],
        ];
        let magnitudes = spec(transform.view());
        assert!((magnitudes[[0, 0]] - 5.0).abs() < 1e-6);
        assert!((magnitudes[[0, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn diff_frames_follow_the_window_shape() {
        // madmom: _diff_frames(0.5, 441, 2048) == 1, _diff_frames(1.0, 441, 2048) == 2
        assert_eq!(diff_frames(0.5, 441.0, 2048, WindowFunction::Hanning), 1);
        assert_eq!(diff_frames(1.0, 441.0, 2048, WindowFunction::Hanning), 2);
        assert_eq!(diff_frames(0.5, 441.0, 1024, WindowFunction::Hanning), 1);
    }

    #[test]
    fn logarithmic_stage_uses_log10_by_default() {
        let spectrogram = Spectrogram::from_values(array![[9.0f32, 99.0]], None);
        let logged = LogarithmicSpectrogramProcessor::default().process(&spectrogram);
        assert!((logged.data()[[0, 0]] - 1.0).abs() < 1e-6); // log10(1 + 9)
        assert!((logged.data()[[0, 1]] - 2.0).abs() < 1e-5); // log10(1 + 99)
    }

    #[test]
    fn difference_zeroes_the_first_frames_and_keeps_positive_values() {
        let spectrogram = Spectrogram::from_values(array![[1.0f32], [3.0], [2.0], [2.0]], None);
        let parameters = DifferenceParameters {
            diff_frames: Some(1),
            positive_diffs: true,
            stack_diffs: StackDiffs::None,
            ..Default::default()
        };
        let difference = spectrogram.difference(&parameters).unwrap();
        let values: Vec<f32> = difference.data().iter().copied().collect();
        // first frame: no predecessor -> 0; then +2 and -1 (clipped to 0) and 0
        assert_eq!(values, vec![0.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn stacking_puts_the_spectrogram_next_to_the_differences() {
        let spectrogram = Spectrogram::from_values(array![[1.0f32], [3.0], [2.0]], None);
        let parameters = DifferenceParameters {
            diff_frames: Some(1),
            stack_diffs: StackDiffs::Horizontal,
            ..Default::default()
        };
        let stacked = spectrogram.difference(&parameters).unwrap();
        assert_eq!(stacked.num_bins(), 2);
        assert_eq!(stacked.num_frames(), 3);
        assert_eq!(stacked.data()[[1, 0]], 3.0); // the spectrogram half
        assert_eq!(stacked.data()[[1, 1]], 2.0); // the difference half
    }

    #[test]
    fn diff_max_bins_is_refused_instead_of_ignored() {
        let spectrogram = Spectrogram::from_values(array![[1.0f32], [2.0]], None);
        let parameters = DifferenceParameters {
            diff_max_bins: Some(3),
            ..Default::default()
        };
        assert!(matches!(
            spectrogram.difference(&parameters),
            Err(SpectrogramError::NotImplemented(_))
        ));
    }
}
