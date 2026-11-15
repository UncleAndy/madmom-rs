//! Tempo estimation: inter-beat-interval histograms and the dominant tempi they contain.
//!
//! Port of `madmom/features/tempo.py`. The chain is short but each step has a detail that
//! changes the answer:
//!
//! 1. the beat activations are smoothed over `act_smooth` seconds (`int(round(fps * 0.14))`
//!    frames — 14 at 100 fps);
//! 2. a histogram over candidate beat intervals is built — with a bank of resonating comb
//!    filters (the default), with autocorrelation, or by counting the intervals of a DBN
//!    path (`comb` / `acf` / `dbn`); a multi-task TCN output comes with its own tempo
//!    classification layer instead (`TCNTempoHistogramProcessor`);
//! 3. the histogram is smoothed over `hist_smooth` bins (a Hamming window of 9);
//! 4. `detect_tempo` takes the local maxima of the histogram, converts delays to bpm when
//!    `fps` is given, and normalises their strengths.
//!
//! Everything is `f64`: the reference casts the activations with `astype(float)` and the
//! strengths are compared against 1e-3 in its own tests.
//!
//! Deferred and refused explicitly: the online (frame-by-frame) mode of the histogram and
//! estimation processors, which needs madmom's `BufferProcessor` streaming semantics; and
//! `interpolate=True`, which needs scipy's quadratic B-spline interpolation
//! (`interp1d(..., 'quadratic')`).

use ndarray::{Array1, Array2, ArrayView1};
use thiserror::Error;

use madmom_audio::comb_filters::{CombFilterFunction, CombFilterbankProcessor};
use madmom_audio::signal::smooth_f64;
use madmom_core::scipy::{argrelmax, Mode, ScipyError};

use crate::beats::DBNBeatTrackingProcessor;

/// Default tempo estimation method (`method='comb'`).
pub const METHOD: &str = "comb";
/// Default scaling factor of the comb filter.
pub const ALPHA: f64 = 0.79;
/// Default minimum tempo [bpm].
pub const MIN_BPM: f64 = 40.0;
/// Default maximum tempo [bpm].
pub const MAX_BPM: f64 = 250.0;
/// Default smoothing of the activation function [seconds].
pub const ACT_SMOOTH: f64 = 0.14;
/// Default smoothing of the tempo histogram [bins].
pub const HIST_SMOOTH: usize = 9;
/// Default aggregation window of the online histogram [seconds].
pub const HIST_BUFFER: f64 = 10.0;
/// Value reported when no tempo was found (madmom's `NO_TEMPO = np.nan`).
pub const NO_TEMPO: f64 = f64::NAN;

/// Errors of the tempo estimation.
#[derive(Debug, Error, PartialEq)]
pub enum TempoError {
    /// Only 1-D activation functions can be used.
    #[error("too many dimensions for the interval histogram: {0}")]
    InvalidDims(usize),
    /// `max_tau` must leave room for at least one delay.
    #[error("no delays left: min_tau {min_tau} against signal length {length}")]
    NoDelays {
        /// Minimal delay asked for.
        min_tau: usize,
        /// Length of the activation function.
        length: usize,
    },
    /// A histogram needs at least one bin.
    #[error("empty histogram")]
    EmptyHistogram,
    /// The underlying `scipy` compatibility layer failed.
    #[error(transparent)]
    Scipy(#[from] ScipyError),
    /// The comb filter bank failed.
    #[error(transparent)]
    CombFilter(#[from] madmom_audio::comb_filters::CombFilterError),
    /// Smoothing the activations or the histogram failed.
    #[error(transparent)]
    Signal(#[from] madmom_audio::signal::SignalError),
    /// `interpolate=True` is not ported (see the module documentation).
    #[error("interpolate=True needs scipy's quadratic B-spline interpolation, not ported yet")]
    InterpolationNotPorted,
}

/// A tempo/interval histogram: the strengths of the bins and their delays.
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    bins: Array1<f64>,
    delays: Array1<f64>,
}

impl Histogram {
    /// Build a histogram from bins and the delays they belong to.
    pub fn new(bins: Array1<f64>, delays: Array1<f64>) -> Result<Self, TempoError> {
        if bins.is_empty() || delays.is_empty() {
            return Err(TempoError::EmptyHistogram);
        }
        assert_eq!(
            bins.len(),
            delays.len(),
            "every bin needs a delay: {} against {}",
            bins.len(),
            delays.len()
        );
        Ok(Self { bins, delays })
    }

    /// The strengths of the bins.
    pub fn bins(&self) -> ArrayView1<'_, f64> {
        self.bins.view()
    }

    /// The delays (inter beat intervals in frames, or tempi in bpm).
    pub fn delays(&self) -> ArrayView1<'_, f64> {
        self.delays.view()
    }
}

/// Smooth the histogram with a Hamming window of `smooth` bins; the delays stay.
pub fn smooth_histogram(histogram: &Histogram, smooth: usize) -> Result<Histogram, TempoError> {
    let bins = smooth_f64(histogram.bins(), smooth)?;
    Ok(Histogram {
        bins,
        delays: histogram.delays.clone(),
    })
}

/// Interval histogram via auto-correlation: `bins[τ] = Σ |x[n] · x[n - τ]|`.
pub fn interval_histogram_acf(
    activations: ArrayView1<'_, f64>,
    min_tau: usize,
    max_tau: Option<usize>,
) -> Result<Histogram, TempoError> {
    let length = activations.len();
    let max_tau = max_tau.unwrap_or_else(|| length.saturating_sub(min_tau));
    if max_tau < min_tau {
        return Err(TempoError::NoDelays { min_tau, length });
    }
    let mut bins = Vec::with_capacity(max_tau - min_tau + 1);
    for tau in min_tau..=max_tau {
        let mut acc = 0.0f64;
        for n in tau..length {
            acc += (activations[n] * activations[n - tau]).abs();
        }
        bins.push(acc);
    }
    Histogram::new(
        Array1::from(bins),
        Array1::from(
            (min_tau..=max_tau)
                .map(|tau| tau as f64)
                .collect::<Vec<_>>(),
        ),
    )
}

/// Interval histogram via a bank of resonating comb filters.
///
/// The reference keeps, for every frame, only the delays that share the maximum response
/// of that frame (`act == np.max(act, axis=-1)[..., np.newaxis]`) and sums those over the
/// frames — so a frame contributes to a single bin per peak, not to every delay.
pub fn interval_histogram_comb(
    activations: ArrayView1<'_, f64>,
    alpha: f64,
    min_tau: usize,
    max_tau: Option<usize>,
) -> Result<Histogram, TempoError> {
    let length = activations.len();
    let max_tau = max_tau.unwrap_or_else(|| length.saturating_sub(min_tau));
    if max_tau < min_tau {
        return Err(TempoError::NoDelays { min_tau, length });
    }
    let taus: Vec<usize> = (min_tau..=max_tau).collect();
    let bank =
        CombFilterbankProcessor::new(CombFilterFunction::Backward, taus.clone(), vec![alpha])?;
    let filtered = bank.process_1d(activations)?;
    let mut bins = vec![0.0f64; taus.len()];
    for frame in 0..filtered.shape()[0] {
        let row = filtered.row(frame);
        let mut maximum = f64::NEG_INFINITY;
        for value in row.iter() {
            if *value > maximum {
                maximum = *value;
            }
        }
        for (bin, value) in row.iter().enumerate() {
            if *value == maximum {
                bins[bin] += *value;
            }
        }
    }
    Histogram::new(
        Array1::from(bins),
        Array1::from(taus.into_iter().map(|tau| tau as f64).collect::<Vec<_>>()),
    )
}

/// The delay of the strongest bin (optionally after smoothing).
pub fn dominant_interval(histogram: &Histogram, smooth: Option<usize>) -> Result<f64, TempoError> {
    let smoothed;
    let histogram = match smooth {
        Some(smooth) if smooth > 0 => {
            smoothed = smooth_histogram(histogram, smooth)?;
            &smoothed
        }
        _ => histogram,
    };
    let mut best = 0usize;
    for (index, value) in histogram.bins().iter().enumerate() {
        if *value > histogram.bins()[best] {
            best = index;
        }
    }
    Ok(histogram.delays()[best])
}

/// The dominant tempi of a histogram, as `(tempo, relative strength)` rows.
///
/// With `fps` given, the delays are interpreted as inter beat intervals in frames and
/// converted to bpm; without it they are already tempi. `interpolate` is refused (see the
/// module documentation). Ties between peaks are ordered like the reference's
/// `argsort(...)[::-1]`: a stable ascending sort by strength, reversed, so the later bin
/// comes first.
pub fn detect_tempo(
    histogram: &Histogram,
    fps: Option<f64>,
    interpolate: bool,
) -> Result<Array2<f64>, TempoError> {
    if interpolate {
        return Err(TempoError::InterpolationNotPorted);
    }
    let mut tempi = histogram.delays.clone();
    let mut bins = histogram.bins.clone();
    if let Some(fps) = fps {
        for value in tempi.iter_mut() {
            *value = 60.0 * fps / *value;
        }
    }
    let peaks = argrelmax(bins.as_slice().expect("a 1-D histogram"), 1, Mode::Wrap)?;
    if peaks.is_empty() {
        return Ok(Array2::from_shape_vec((1, 2), vec![NO_TEMPO, 0.0]).expect("1x2"));
    }
    if peaks.len() == 1 {
        return Ok(Array2::from_shape_vec((1, 2), vec![tempi[peaks[0]], 1.0]).expect("1x2"));
    }
    let mut order: Vec<usize> = (0..peaks.len()).collect();
    // stable ascending sort by strength, then reversed: equal strengths end up with the
    // later peak first, which is what `argsort(...)[::-1]` produces
    order.sort_by(|left, right| {
        bins[peaks[*left]]
            .partial_cmp(&bins[peaks[*right]])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.reverse();
    let total: f64 = order.iter().map(|index| bins[peaks[*index]]).sum();
    let mut out = Array2::<f64>::zeros((order.len(), 2));
    for (row, index) in order.iter().enumerate() {
        out[[row, 0]] = tempi[peaks[*index]];
        out[[row, 1]] = bins[peaks[*index]] / total;
    }
    let _ = &mut bins;
    Ok(out)
}

/// Anything that turns data into a tempo histogram (madmom's `TempoHistogramProcessor`).
pub trait TempoHistogramSource {
    /// Minimum tempo [bpm].
    fn min_bpm(&self) -> f64;
    /// Maximum tempo [bpm].
    fn max_bpm(&self) -> f64;
    /// Frame rate, if the source works on frames.
    fn fps(&self) -> Option<f64>;
    /// Minimum beat interval [frames] (`floor(60 * fps / max_bpm)`).
    fn min_interval(&self) -> usize;
    /// Maximum beat interval [frames] (`ceil(60 * fps / min_bpm)`).
    fn max_interval(&self) -> usize;
    /// The candidate beat intervals [frames].
    fn intervals(&self) -> Array1<f64> {
        Array1::from(
            (self.min_interval()..=self.max_interval())
                .map(|interval| interval as f64)
                .collect::<Vec<_>>(),
        )
    }
    /// Build the histogram of the given data (activations, or TCN output for the TCN one).
    fn histogram(&self, data: ArrayView1<'_, f64>) -> Result<Histogram, TempoError>;
}

/// Frame-based histogram source: the interval bounds every frame-based variant shares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntervalBounds {
    min_bpm: f64,
    max_bpm: f64,
    fps: f64,
}

impl IntervalBounds {
    /// Build the bounds.
    pub fn new(min_bpm: f64, max_bpm: f64, fps: f64) -> Self {
        Self {
            min_bpm,
            max_bpm,
            fps,
        }
    }
}

impl TempoHistogramSource for IntervalBounds {
    fn min_bpm(&self) -> f64 {
        self.min_bpm
    }

    fn max_bpm(&self) -> f64 {
        self.max_bpm
    }

    fn fps(&self) -> Option<f64> {
        Some(self.fps)
    }

    fn min_interval(&self) -> usize {
        (60.0 * self.fps / self.max_bpm).floor() as usize
    }

    fn max_interval(&self) -> usize {
        (60.0 * self.fps / self.min_bpm).ceil() as usize
    }

    fn histogram(&self, _data: ArrayView1<'_, f64>) -> Result<Histogram, TempoError> {
        unreachable!("the bounds only describe the interval range")
    }
}

/// Tempo histogram built with a bank of resonating comb filters (the default method).
#[derive(Debug, Clone, PartialEq)]
pub struct CombFilterTempoHistogramProcessor {
    bounds: IntervalBounds,
    alpha: f64,
    hist_buffer: f64,
}

impl CombFilterTempoHistogramProcessor {
    /// Build the processor.
    pub fn new(min_bpm: f64, max_bpm: f64, alpha: f64, hist_buffer: f64, fps: f64) -> Self {
        Self {
            bounds: IntervalBounds::new(min_bpm, max_bpm, fps),
            alpha,
            hist_buffer,
        }
    }

    /// Build the processor with the reference's defaults.
    pub fn with_defaults(fps: f64) -> Self {
        Self::new(MIN_BPM, MAX_BPM, ALPHA, HIST_BUFFER, fps)
    }

    /// The scaling factor of the comb filter.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// The online aggregation window [seconds].
    pub fn hist_buffer(&self) -> f64 {
        self.hist_buffer
    }
}

impl TempoHistogramSource for CombFilterTempoHistogramProcessor {
    fn min_bpm(&self) -> f64 {
        self.bounds.min_bpm()
    }

    fn max_bpm(&self) -> f64 {
        self.bounds.max_bpm()
    }

    fn fps(&self) -> Option<f64> {
        self.bounds.fps()
    }

    fn min_interval(&self) -> usize {
        self.bounds.min_interval()
    }

    fn max_interval(&self) -> usize {
        self.bounds.max_interval()
    }

    fn histogram(&self, data: ArrayView1<'_, f64>) -> Result<Histogram, TempoError> {
        interval_histogram_comb(
            data,
            self.alpha,
            self.min_interval(),
            Some(self.max_interval()),
        )
    }
}

/// Tempo histogram built by auto-correlating the activations.
#[derive(Debug, Clone, PartialEq)]
pub struct ACFTempoHistogramProcessor {
    bounds: IntervalBounds,
    hist_buffer: f64,
}

impl ACFTempoHistogramProcessor {
    /// Build the processor.
    pub fn new(min_bpm: f64, max_bpm: f64, hist_buffer: f64, fps: f64) -> Self {
        Self {
            bounds: IntervalBounds::new(min_bpm, max_bpm, fps),
            hist_buffer,
        }
    }

    /// Build the processor with the reference's defaults.
    pub fn with_defaults(fps: f64) -> Self {
        Self::new(MIN_BPM, MAX_BPM, HIST_BUFFER, fps)
    }

    /// The online aggregation window [seconds].
    pub fn hist_buffer(&self) -> f64 {
        self.hist_buffer
    }
}

impl TempoHistogramSource for ACFTempoHistogramProcessor {
    fn min_bpm(&self) -> f64 {
        self.bounds.min_bpm()
    }

    fn max_bpm(&self) -> f64 {
        self.bounds.max_bpm()
    }

    fn fps(&self) -> Option<f64> {
        self.bounds.fps()
    }

    fn min_interval(&self) -> usize {
        self.bounds.min_interval()
    }

    fn max_interval(&self) -> usize {
        self.bounds.max_interval()
    }

    fn histogram(&self, data: ArrayView1<'_, f64>) -> Result<Histogram, TempoError> {
        interval_histogram_acf(data, self.min_interval(), Some(self.max_interval()))
    }
}

/// Tempo histogram from the intervals of a DBN beat-tracking path.
///
/// The reference decodes the activations cast to **float32** (`astype(np.float32)`), counts
/// how often each state interval occurs along the path and drops the bins below the state
/// space's smallest interval.
#[derive(Debug, Clone)]
pub struct DBNTempoHistogramProcessor {
    bounds: IntervalBounds,
    hist_buffer: f64,
    dbn: DBNBeatTrackingProcessor,
}

impl DBNTempoHistogramProcessor {
    /// Build the processor.
    pub fn new(min_bpm: f64, max_bpm: f64, hist_buffer: f64, fps: f64) -> Self {
        let dbn = DBNBeatTrackingProcessor::new(
            min_bpm,
            max_bpm,
            None,
            crate::beats::TRANSITION_LAMBDA,
            crate::beats::OBSERVATION_LAMBDA,
            crate::beats::THRESHOLD,
            crate::beats::CORRECT,
            fps,
        );
        Self {
            bounds: IntervalBounds::new(min_bpm, max_bpm, fps),
            hist_buffer,
            dbn,
        }
    }

    /// Build the processor with the reference's defaults.
    pub fn with_defaults(fps: f64) -> Self {
        Self::new(MIN_BPM, MAX_BPM, HIST_BUFFER, fps)
    }

    /// The beat tracker whose path provides the intervals.
    pub fn dbn(&self) -> &DBNBeatTrackingProcessor {
        &self.dbn
    }

    /// The online aggregation window [seconds].
    pub fn hist_buffer(&self) -> f64 {
        self.hist_buffer
    }
}

impl TempoHistogramSource for DBNTempoHistogramProcessor {
    fn min_bpm(&self) -> f64 {
        self.bounds.min_bpm()
    }

    fn max_bpm(&self) -> f64 {
        self.bounds.max_bpm()
    }

    fn fps(&self) -> Option<f64> {
        self.bounds.fps()
    }

    fn min_interval(&self) -> usize {
        self.bounds.min_interval()
    }

    fn max_interval(&self) -> usize {
        self.bounds.max_interval()
    }

    fn intervals(&self) -> Array1<f64> {
        let intervals = self.dbn.state_space().intervals();
        Array1::from(
            intervals
                .iter()
                .map(|value| *value as f64)
                .collect::<Vec<_>>(),
        )
    }

    fn histogram(&self, data: ArrayView1<'_, f64>) -> Result<Histogram, TempoError> {
        let state_space = self.dbn.state_space();
        let intervals = state_space.intervals();
        // the reference casts the activations to float32 before decoding
        let activations: Vec<f64> = data.iter().map(|value| (*value as f32) as f64).collect();
        let matrix = Array2::from_shape_vec((activations.len(), 1), activations)
            .expect("one column holds all activations");
        let result = self.dbn.hmm().viterbi(matrix.view());
        let maximum = *intervals.iter().max().expect("intervals are not empty");
        let minimum = *intervals.iter().min().expect("intervals are not empty");
        let mut counts = vec![0.0f64; maximum + 1];
        for state in &result.path {
            let interval = state_space.state_intervals()[*state as usize];
            counts[interval] += 1.0;
        }
        let bins = counts[minimum..].to_vec();
        Histogram::new(
            Array1::from(bins),
            Array1::from(
                intervals
                    .iter()
                    .map(|value| *value as f64)
                    .collect::<Vec<_>>(),
            ),
        )
    }
}

/// Tempo histogram taken from the tempo classification layer of a multi-task TCN.
///
/// The delays are tempi in bpm (a linear range, one bin per bpm), and the range is cut to
/// `min_bpm..=max_bpm`.
#[derive(Debug, Clone, PartialEq)]
pub struct TCNTempoHistogramProcessor {
    min_bpm: f64,
    max_bpm: f64,
}

impl TCNTempoHistogramProcessor {
    /// Build the processor.
    pub fn new(min_bpm: f64, max_bpm: f64) -> Self {
        Self { min_bpm, max_bpm }
    }

    /// Build the processor with the reference's defaults.
    pub fn with_defaults() -> Self {
        Self::new(MIN_BPM, MAX_BPM)
    }
}

impl TempoHistogramSource for TCNTempoHistogramProcessor {
    fn min_bpm(&self) -> f64 {
        self.min_bpm
    }

    fn max_bpm(&self) -> f64 {
        self.max_bpm
    }

    fn fps(&self) -> Option<f64> {
        // a TCN's tempo layer is already in bpm
        None
    }

    fn min_interval(&self) -> usize {
        self.min_bpm as usize
    }

    fn max_interval(&self) -> usize {
        self.max_bpm as usize
    }

    /// The tempo range is the TCN's own bin index, i.e. `arange(len(data))` filtered to
    /// `min_bpm..=max_bpm` — not the beat-interval range of the frame-based sources.
    fn intervals(&self) -> Array1<f64> {
        Array1::from(
            (self.min_interval()..=self.max_interval())
                .map(|tempo| tempo as f64)
                .collect::<Vec<_>>(),
        )
    }

    fn histogram(&self, data: ArrayView1<'_, f64>) -> Result<Histogram, TempoError> {
        // tempi = arange(len(data)); min_idx = argmax(tempi >= min_bpm);
        // max_idx = argmin(tempi <= max_bpm) -> the first bin above max_bpm
        let min_index = (0..data.len())
            .find(|index| *index as f64 >= self.min_bpm)
            .unwrap_or(data.len());
        let max_index = (0..data.len())
            .find(|index| (*index as f64) > self.max_bpm)
            .unwrap_or(data.len());
        if max_index <= min_index {
            return Err(TempoError::EmptyHistogram);
        }
        let bins = Array1::from(data.slice(ndarray::s![min_index..max_index]).to_vec());
        let delays = Array1::from(
            (min_index..max_index)
                .map(|index| index as f64)
                .collect::<Vec<_>>(),
        );
        Histogram::new(bins, delays)
    }
}

/// Estimate the dominant tempi of an activation function (or TCN output).
#[derive(Debug, Clone)]
pub struct TempoEstimationProcessor<S: TempoHistogramSource> {
    histogram_processor: S,
    act_smooth: Option<f64>,
    hist_smooth: usize,
    fps: Option<f64>,
    interpolate: bool,
}

impl<S: TempoHistogramSource> TempoEstimationProcessor<S> {
    /// Build the processor from a histogram source.
    ///
    /// `act_smooth` is the smoothing of the activations in seconds; the reference sets it
    /// to `None` for the DBN and TCN sources.
    pub fn new(
        histogram_processor: S,
        act_smooth: Option<f64>,
        hist_smooth: usize,
        fps: Option<f64>,
        interpolate: bool,
    ) -> Self {
        Self {
            histogram_processor,
            act_smooth,
            hist_smooth,
            fps,
            interpolate,
        }
    }

    /// Build the processor with the reference's defaults for the comb method.
    pub fn with_defaults(histogram_processor: S, fps: f64) -> Self {
        Self::new(
            histogram_processor,
            Some(ACT_SMOOTH),
            HIST_SMOOTH,
            Some(fps),
            false,
        )
    }

    /// The histogram source.
    pub fn histogram_processor(&self) -> &S {
        &self.histogram_processor
    }

    /// Smoothing of the activations [seconds], if any.
    pub fn act_smooth(&self) -> Option<f64> {
        self.act_smooth
    }

    /// Smoothing of the histogram [bins].
    pub fn hist_smooth(&self) -> usize {
        self.hist_smooth
    }

    /// The frame rate, if the source is frame based.
    pub fn fps(&self) -> Option<f64> {
        self.fps
    }

    /// The histogram of the (smoothed) data, without detecting the tempi.
    pub fn histogram(&self, data: ArrayView1<'_, f64>) -> Result<Histogram, TempoError> {
        let smoothed;
        let data = match (self.act_smooth, self.fps) {
            (Some(seconds), Some(fps)) => {
                let size = (fps * seconds).round() as usize;
                smoothed = smooth_f64(data, size)?;
                smoothed.view()
            }
            _ => data,
        };
        self.histogram_processor.histogram(data)
    }

    /// Detect the tempi of the given data.
    pub fn process(&self, data: ArrayView1<'_, f64>) -> Result<Array2<f64>, TempoError> {
        let histogram = self.histogram(data)?;
        let histogram = smooth_histogram(&histogram, self.hist_smooth)?;
        detect_tempo(&histogram, self.fps, self.interpolate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn acf_histogram_matches_the_reference_definition() {
        // a perfect pulse train at period 4: the correlation peaks at multiples of 4
        let mut activations = vec![0.0f64; 20];
        for frame in (0..20).step_by(4) {
            activations[frame] = 1.0;
        }
        let histogram = interval_histogram_acf(ArrayView1::from(&activations), 1, Some(8)).unwrap();
        assert_eq!(
            histogram.delays().to_vec(),
            (1..=8).map(|t| t as f64).collect::<Vec<_>>()
        );
        // five pulses at 0, 4, 8, 12, 16: tau = 4 pairs them four times, tau = 8 three
        // times (0-8, 4-12, 8-16)
        assert_eq!(histogram.bins()[3], 4.0);
        assert_eq!(histogram.bins()[7], 3.0);
    }

    #[test]
    fn comb_histogram_counts_only_the_winning_delay_per_frame() {
        let mut activations = vec![0.0f64; 32];
        for frame in (0..32).step_by(8) {
            activations[frame] = 1.0;
        }
        let histogram =
            interval_histogram_comb(ArrayView1::from(&activations), 0.79, 4, Some(16)).unwrap();
        let bins = histogram.bins();
        let delays = histogram.delays();
        let best = (0..bins.len())
            .max_by(|left, right| bins[*left].partial_cmp(&bins[*right]).unwrap())
            .unwrap();
        assert_eq!(
            delays[best], 8.0,
            "the resonance must sit at the pulse period"
        );
    }

    #[test]
    fn detect_tempo_normalises_the_strengths_and_refuses_interpolation() {
        let histogram = Histogram::new(
            array![0.0, 5.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            Array1::from((1..=9).map(|value| value as f64).collect::<Vec<_>>()),
        )
        .unwrap();
        let tempi = detect_tempo(&histogram, None, false).unwrap();
        assert_eq!(tempi.shape(), &[3, 2]);
        assert_eq!(tempi[[0, 0]], 2.0);
        assert!((tempi[[0, 1]] - 0.5).abs() < 1e-12);
        assert!((tempi[[1, 1]] - 0.3).abs() < 1e-12);
        assert!((tempi[[2, 1]] - 0.2).abs() < 1e-12);
        assert_eq!(
            detect_tempo(&histogram, None, true),
            Err(TempoError::InterpolationNotPorted)
        );
    }

    #[test]
    fn a_flat_histogram_has_no_tempo() {
        let histogram = Histogram::new(
            Array1::from(vec![1.0; 5]),
            Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        )
        .unwrap();
        let tempi = detect_tempo(&histogram, None, false).unwrap();
        assert_eq!(tempi.shape(), &[1, 2]);
        assert!(tempi[[0, 0]].is_nan());
        assert_eq!(tempi[[0, 1]], 0.0);
    }

    #[test]
    fn interval_bounds_follow_the_reference_rounding() {
        // 40..250 bpm at 100 fps -> floor(60*100/250)=24, ceil(60*100/40)=150
        let bounds = IntervalBounds::new(MIN_BPM, MAX_BPM, 100.0);
        assert_eq!(bounds.min_interval(), 24);
        assert_eq!(bounds.max_interval(), 150);
        assert_eq!(bounds.intervals().len(), 127);
    }

    #[test]
    fn tcn_histogram_cuts_the_tempo_range() {
        let processor = TCNTempoHistogramProcessor::new(10.0, 250.0);
        let data: Vec<f64> = (0..300).map(|index| index as f64).collect();
        let histogram = processor.histogram(ArrayView1::from(&data)).unwrap();
        assert_eq!(histogram.delays()[0], 10.0);
        assert_eq!(histogram.delays()[histogram.delays().len() - 1], 250.0);
        assert_eq!(histogram.bins().len(), 241);
    }
}
