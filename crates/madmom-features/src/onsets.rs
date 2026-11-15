//! Onset detection functions and peak picking.
//!
//! Port of `madmom/features/onsets.py`: the eleven onset detection functions (ODFs) that
//! turn a spectrogram into an activation function, and the peak picker that turns the
//! activations into onset times.
//!
//! The ODFs differ in what they look at — the magnitudes, their first-order difference, the
//! phase, or the complex spectrum — and every one of them has a detail that decides the
//! numbers:
//!
//! * `superflux` and `complex_flux` take the difference against a **maximum filtered**
//!   spectrogram (`diff_max_bins`, three bins by default), which is what suppresses
//!   vibrato: a peak that only moved one bin produces no difference at all;
//! * `complex_flux` weights that difference with a mask built from the **local group
//!   delay** of the phase (needs `circular_shift=True` on the STFT), expanded to the
//!   corner bins of each filterbank band;
//! * `modified_kullback_leibler` divides by the previous frame plus `np.spacing(1)`, not by
//!   zero, and averages the logarithm — the original MKL sums;
//! * the phase ODFs use second-order phase differences wrapped to `-pi..pi`, and
//!   `rectified_complex_domain` keeps only the bins whose magnitude rose.
//!
//! Phase-based ODFs need the transform the spectrogram was computed from, so they take the
//! phase matrix explicitly (the reference reaches it as `spectrogram.stft.phase()`).
//!
//! Not ported here: the ODF *processors* that wrap a network (`RNNOnsetProcessor`,
//! `CNNOnsetProcessor`, `SpectralOnsetProcessor`) — they belong with the inference phase,
//! where the activation model lives.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex64;
use thiserror::Error;

use madmom_audio::signal::{smooth_f64, SignalError};
use madmom_audio::spectrogram::{DifferenceParameters, Spectrogram, SpectrogramError};
use madmom_audio::stft::local_group_delay;
use madmom_core::ndimage::{
    maximum_filter_1d, maximum_filter_2d, minimum_filter_2d, uniform_filter_1d, Mode,
};
use madmom_core::utils::{combine_events, Combine};

/// The epsilon the reference adds before dividing (`np.spacing(1)`).
pub const EPSILON: f64 = f64::EPSILON;

/// Errors of the onset detection functions.
#[derive(Debug, Error)]
pub enum OnsetError {
    /// Only 1-D activation functions can be peak picked.
    #[error("`activations` must be either 1D or 2D, got {0} dimensions")]
    InvalidDims(usize),
    /// `epsilon` must be positive (the reference raises here).
    #[error("a positive value must be added before division, got {0}")]
    InvalidEpsilon(f64),
    /// `diff_frames` must be at least 1.
    #[error("number of `diff_frames` must be >= 1, got {0}")]
    InvalidDiffFrames(usize),
    /// The spectrogram and the phase must have the same shape.
    #[error("spectrogram and phase must be of same shape: {spectrogram:?} against {phase:?}")]
    ShapeMismatch {
        /// Shape of the spectrogram.
        spectrogram: (usize, usize),
        /// Shape of the phase.
        phase: (usize, usize),
    },
    /// The reference refuses a `delay` in online mode.
    #[error("delay not supported yet in online mode")]
    DelayInOnlineMode,
    /// A difference stage failed.
    #[error(transparent)]
    Spectrogram(#[from] SpectrogramError),
    /// Smoothing failed.
    #[error(transparent)]
    Signal(#[from] SignalError),
}

/// Wrap phase values to the range `-pi..pi` (`np.mod(phase + pi, 2*pi) - pi`).
pub fn wrap_to_pi(values: ArrayView2<'_, f64>) -> Array2<f64> {
    values.mapv(|value| {
        (value + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
    })
}

/// Second-order phase difference wrapped to `-pi..pi` (`_phase_deviation`).
pub fn phase_deviation_2d(phase: ArrayView2<'_, f64>) -> Array2<f64> {
    let (rows, columns) = (phase.shape()[0], phase.shape()[1]);
    let mut deviation = Array2::<f64>::zeros((rows, columns));
    if rows < 3 {
        return deviation;
    }
    for row in 2..rows {
        for column in 0..columns {
            deviation[[row, column]] =
                phase[[row, column]] - 2.0 * phase[[row - 1, column]] + phase[[row - 2, column]];
        }
    }
    wrap_to_pi(deviation.view())
}

/// Complex domain of the spectrogram (`_complex_domain`).
///
/// The target assumes a constant phase change per frame: `2*phase[n] - phase[n-1]`, with
/// the first frame's target being zero.
pub fn complex_domain_2d(
    spectrogram: &Spectrogram,
    phase: ArrayView2<'_, f64>,
) -> Result<Array2<Complex64>, OnsetError> {
    let (rows, columns) = (spectrogram.num_frames(), spectrogram.num_bins());
    if phase.shape() != [rows, columns] {
        return Err(OnsetError::ShapeMismatch {
            spectrogram: (rows, columns),
            phase: (phase.shape()[0], phase.shape()[1]),
        });
    }
    let magnitudes = spectrogram.data();
    let mut target = Array2::<Complex64>::zeros((rows, columns));
    for row in 1..rows {
        for column in 0..columns {
            let angle = 2.0 * phase[[row, column]] - phase[[row - 1, column]];
            target[[row, column]] = Complex64::from_polar(magnitudes[[row, column]] as f64, angle);
        }
    }
    // the target phase of the first frame is zero (`cd_target[0]` is never assigned), so
    // that frame's target is the bare magnitude — not the magnitude with its own phase
    for column in 0..columns {
        target[[0, column]] = Complex64::new(magnitudes[[0, column]] as f64, 0.0);
    }
    let mut domain = Array2::<Complex64>::zeros((rows, columns));
    for row in 0..rows {
        for column in 0..columns {
            domain[[row, column]] =
                Complex64::from_polar(magnitudes[[row, column]] as f64, phase[[row, column]]);
        }
    }
    for row in 1..rows {
        for column in 0..columns {
            domain[[row, column]] -= target[[row - 1, column]];
        }
    }
    Ok(domain)
}

fn mean_over_bins(values: ArrayView2<'_, f64>) -> Array1<f64> {
    let columns = values.shape()[1].max(1) as f64;
    values.map_axis(ndarray::Axis(1), |row| row.sum() / columns)
}

fn sum_over_bins(values: ArrayView2<'_, f64>) -> Array1<f64> {
    values.map_axis(ndarray::Axis(1), |row| row.sum())
}

/// High frequency content: the magnitudes weighted by their bin number.
pub fn high_frequency_content(spectrogram: &Spectrogram) -> Array1<f64> {
    let bins = spectrogram.num_bins();
    let weighted = spectrogram.data().mapv(|value| value as f64)
        * Array2::<f64>::from_shape_fn((1, bins), |(_, bin)| bin as f64);
    mean_over_bins(weighted.view())
}

/// The difference of the spectrogram the ODFs need, unless it already is a difference.
fn differences(
    spectrogram: &Spectrogram,
    diff_frames: Option<usize>,
    diff_max_bins: Option<usize>,
) -> Result<Spectrogram, OnsetError> {
    if let Some(parameters) = spectrogram.difference_parameters() {
        if parameters.positive_diffs {
            return Ok(spectrogram.clone());
        }
    }
    Ok(spectrogram.difference(&DifferenceParameters {
        diff_frames,
        diff_max_bins,
        positive_diffs: true,
        ..Default::default()
    })?)
}

/// Spectral diff: the sum of all squared positive first-order differences.
pub fn spectral_diff(
    spectrogram: &Spectrogram,
    diff_frames: Option<usize>,
) -> Result<Array1<f64>, OnsetError> {
    let differences = differences(spectrogram, diff_frames, None)?;
    let squared = differences
        .data()
        .mapv(|value| (value as f64) * (value as f64));
    Ok(sum_over_bins(squared.view()))
}

/// Spectral flux: the sum of all positive first-order differences.
pub fn spectral_flux(
    spectrogram: &Spectrogram,
    diff_frames: Option<usize>,
) -> Result<Array1<f64>, OnsetError> {
    let differences = differences(spectrogram, diff_frames, None)?;
    Ok(sum_over_bins(
        differences.data().mapv(|value| value as f64).view(),
    ))
}

/// SuperFlux: the sum of all positive differences to a maximum filtered spectrogram.
pub fn superflux(
    spectrogram: &Spectrogram,
    diff_frames: Option<usize>,
    diff_max_bins: usize,
) -> Result<Array1<f64>, OnsetError> {
    let differences = differences(spectrogram, diff_frames, Some(diff_max_bins))?;
    Ok(sum_over_bins(
        differences.data().mapv(|value| value as f64).view(),
    ))
}

/// ComplexFlux: SuperFlux weighted by a local-group-delay mask.
pub fn complex_flux(
    spectrogram: &Spectrogram,
    phase: ArrayView2<'_, f64>,
    diff_frames: Option<usize>,
    diff_max_bins: usize,
    temporal_filter: usize,
    temporal_origin: i64,
) -> Result<Array1<f64>, OnsetError> {
    // note: the phase does NOT have to match the spectrogram's shape here. The reference
    // reaches the phase through `spectrogram.stft`, i.e. the transform of the *unfiltered*
    // spectrogram, and uses it at full bin resolution while the mask is built per filterbank
    // band — so a filtered spectrogram with a 1024-bin phase is the normal case.
    let (rows, columns) = (spectrogram.num_frames(), spectrogram.num_bins());
    let phase_bins = phase.shape()[1];
    // absolute local group delay, normalised to 0..1
    let mut group_delay = local_group_delay(phase).mapv(|value| value.abs() / std::f64::consts::PI);
    if temporal_filter > 0 {
        // the reference passes a scalar origin, which scipy broadcasts to both axes; with
        // a size of 1 on the second axis a non-zero origin shifts that column
        group_delay = maximum_filter_2d(
            group_delay.view(),
            (temporal_filter, 1),
            (temporal_origin, temporal_origin),
            Mode::Reflect,
        );
    }
    let mut mask = Array2::<f64>::zeros((rows, columns));
    match spectrogram.filterbank() {
        Some(filterbank) => {
            for band in 0..columns {
                let (first, last) = filterbank.band_extent(band);
                let start = first.saturating_sub(1);
                let stop = (last + 2).min(phase_bins);
                for row in 0..rows {
                    let mut minimum = f64::INFINITY;
                    for column in start..stop {
                        let value = group_delay[[row, column]];
                        if value < minimum {
                            minimum = value;
                        }
                    }
                    mask[[row, band]] = minimum;
                }
            }
        }
        // the reference falls back to a plain minimum filter when the spectrogram was not
        // filtered (it catches the resulting AttributeError)
        None => {
            mask = minimum_filter_2d(group_delay.view(), (1, 3), (0, 0), Mode::Reflect);
        }
    }
    let differences = differences(spectrogram, diff_frames, Some(diff_max_bins))?;
    let weighted = differences.data().mapv(|value| value as f64) * mask;
    Ok(sum_over_bins(weighted.view()))
}

/// Modified Kullback-Leibler: the mean of `log(1 + x[n] / (x[n-1] + epsilon))`.
pub fn modified_kullback_leibler(
    spectrogram: &Spectrogram,
    diff_frames: usize,
    epsilon: f64,
) -> Result<Array1<f64>, OnsetError> {
    if diff_frames < 1 {
        return Err(OnsetError::InvalidDiffFrames(diff_frames));
    }
    if epsilon <= 0.0 {
        return Err(OnsetError::InvalidEpsilon(epsilon));
    }
    let (rows, columns) = (spectrogram.num_frames(), spectrogram.num_bins());
    let magnitudes = spectrogram.data();
    let mut mkl = Array2::<f64>::zeros((rows, columns));
    for row in diff_frames..rows {
        for column in 0..columns {
            mkl[[row, column]] = (magnitudes[[row, column]] as f64)
                / (magnitudes[[row - diff_frames, column]] as f64 + epsilon);
        }
    }
    let logged = mkl.mapv(|value| (1.0 + value).ln());
    Ok(mean_over_bins(logged.view()))
}

/// Phase deviation: the mean absolute change of the instantaneous frequency.
pub fn phase_deviation(
    _spectrogram: &Spectrogram,
    phase: ArrayView2<'_, f64>,
) -> Result<Array1<f64>, OnsetError> {
    // like the reference, this ODF looks only at the phase — which for a filtered
    // spectrogram still has the bins of the unfiltered one, and the mean is taken over
    // those. The spectrogram argument is kept for symmetry with the other ODFs.
    let deviation = phase_deviation_2d(phase).mapv(|value| value.abs());
    Ok(mean_over_bins(deviation.view()))
}

/// Weighted phase deviation: the phase deviation weighted by the magnitudes.
pub fn weighted_phase_deviation(
    spectrogram: &Spectrogram,
    phase: ArrayView2<'_, f64>,
) -> Result<Array1<f64>, OnsetError> {
    let (rows, columns) = (spectrogram.num_frames(), spectrogram.num_bins());
    if phase.shape() != [rows, columns] {
        return Err(OnsetError::ShapeMismatch {
            spectrogram: (rows, columns),
            phase: (rows, columns),
        });
    }
    let deviation = phase_deviation_2d(phase);
    let weighted = deviation * spectrogram.data().mapv(|value| value as f64);
    Ok(mean_over_bins(weighted.mapv(|value| value.abs()).view()))
}

/// Normalized weighted phase deviation: WPD divided by the mean magnitude plus epsilon.
pub fn normalized_weighted_phase_deviation(
    spectrogram: &Spectrogram,
    phase: ArrayView2<'_, f64>,
    epsilon: f64,
) -> Result<Array1<f64>, OnsetError> {
    if epsilon <= 0.0 {
        return Err(OnsetError::InvalidEpsilon(epsilon));
    }
    let weighted = weighted_phase_deviation(spectrogram, phase)?;
    let magnitudes = spectrogram.data().mapv(|value| value as f64);
    let norm = mean_over_bins(magnitudes.view()).mapv(|value| value + epsilon);
    Ok(weighted / norm)
}

/// Complex domain: the sum of the absolute changes of the complex spectrum.
pub fn complex_domain(
    spectrogram: &Spectrogram,
    phase: ArrayView2<'_, f64>,
) -> Result<Array1<f64>, OnsetError> {
    let domain = complex_domain_2d(spectrogram, phase)?;
    let magnitudes = domain.mapv(|value| value.norm());
    Ok(sum_over_bins(magnitudes.view()))
}

/// Rectified complex domain: the complex domain, kept only where the magnitude rose.
pub fn rectified_complex_domain(
    spectrogram: &Spectrogram,
    phase: ArrayView2<'_, f64>,
    diff_frames: Option<usize>,
) -> Result<Array1<f64>, OnsetError> {
    let domain = complex_domain_2d(spectrogram, phase)?;
    let positive = differences(spectrogram, diff_frames, None)?;
    let magnitudes = Array2::<f64>::from_shape_fn(domain.raw_dim(), |(row, column)| {
        if positive.data()[[row, column]] != 0.0 {
            domain[[row, column]].norm()
        } else {
            0.0
        }
    });
    Ok(sum_over_bins(magnitudes.view()))
}

/// Pick the peaks of an activation function.
///
/// `smooth`, `pre_avg`, `post_avg`, `pre_max` and `post_max` are given in **frames** here
/// (the processor converts its seconds before calling). The moving average and the moving
/// maximum use `mode='constant'` — samples outside the activations count as zero, but the
/// window size still divides — and their origins follow the reference's
/// `floor((pre - post) / 2)`.
pub fn peak_picking(
    activations: ArrayView1<'_, f64>,
    threshold: f64,
    smooth: Option<usize>,
    pre_avg: usize,
    post_avg: usize,
    pre_max: usize,
    post_max: usize,
) -> Result<Vec<usize>, OnsetError> {
    let activations = match smooth {
        Some(size) => smooth_f64(activations, size)?,
        None => activations.to_owned(),
    };
    let average_length = pre_avg + post_avg + 1;
    let moving_average = if average_length > 1 {
        let origin = ((pre_avg as f64 - post_avg as f64) / 2.0).floor() as i64;
        Some(uniform_filter_1d(
            activations.view(),
            average_length,
            origin,
            Mode::constant(),
        ))
    } else {
        None
    };
    let mut detections = Array1::<f64>::zeros(activations.len());
    for (frame, activation) in activations.iter().enumerate() {
        let average = moving_average
            .as_ref()
            .map(|average| average[frame])
            .unwrap_or(0.0);
        if *activation >= average + threshold {
            detections[frame] = *activation;
        }
    }
    let maximum_length = pre_max + post_max + 1;
    if maximum_length > 1 {
        let origin = ((pre_max as f64 - post_max as f64) / 2.0).floor() as i64;
        let moving_maximum =
            maximum_filter_1d(detections.view(), maximum_length, origin, Mode::constant());
        for frame in 0..detections.len() {
            if detections[frame] != moving_maximum[frame] {
                detections[frame] = 0.0;
            }
        }
    }
    Ok(detections
        .iter()
        .enumerate()
        .filter(|(_, value)| **value != 0.0)
        .map(|(frame, _)| frame)
        .collect())
}

/// Peak picking of an activation function, with the reference's constants.
#[derive(Debug, Clone, PartialEq)]
pub struct OnsetPeakPickingProcessor {
    threshold: f64,
    smooth: f64,
    pre_avg: f64,
    post_avg: f64,
    pre_max: f64,
    post_max: f64,
    combine: f64,
    delay: f64,
    fps: f64,
    online: bool,
    // online state
    buffer: Option<Vec<f64>>,
    counter: i64,
    last_onset: Option<f64>,
}

/// Default frame rate of the onset peak picker.
pub const FPS: f64 = 100.0;
/// Default detection threshold (a binary threshold in the reference's words).
pub const THRESHOLD: f64 = 0.5;
/// Default smoothing of the activations [seconds].
pub const SMOOTH: f64 = 0.0;
/// Default averaging window before the detections [seconds].
pub const PRE_AVG: f64 = 0.0;
/// Default averaging window after the detections [seconds].
pub const POST_AVG: f64 = 0.0;
/// Default maximum search window before the detections [seconds].
pub const PRE_MAX: f64 = 0.0;
/// Default maximum search window after the detections [seconds].
pub const POST_MAX: f64 = 0.0;
/// Default combination of onsets closer than this [seconds].
pub const COMBINE: f64 = 0.03;
/// Default delay of the reported onsets [seconds].
pub const DELAY: f64 = 0.0;

impl Default for OnsetPeakPickingProcessor {
    fn default() -> Self {
        Self::new(
            THRESHOLD, SMOOTH, PRE_AVG, POST_AVG, PRE_MAX, POST_MAX, COMBINE, DELAY, FPS, false,
        )
    }
}

impl OnsetPeakPickingProcessor {
    /// Build the processor.
    ///
    /// In online mode the reference forces `smooth`, `post_avg` and `post_max` to zero,
    /// because no future information is available; that is done here as well.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        threshold: f64,
        smooth: f64,
        pre_avg: f64,
        post_avg: f64,
        pre_max: f64,
        post_max: f64,
        combine: f64,
        delay: f64,
        fps: f64,
        online: bool,
    ) -> Self {
        let (smooth, post_avg, post_max) = if online {
            (0.0, 0.0, 0.0)
        } else {
            (smooth, post_avg, post_max)
        };
        Self {
            threshold,
            smooth,
            pre_avg,
            post_avg,
            pre_max,
            post_max,
            combine,
            delay,
            fps,
            online,
            buffer: None,
            counter: 0,
            last_onset: None,
        }
    }

    /// The detection threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Whether the processor works causally.
    pub fn is_online(&self) -> bool {
        self.online
    }

    /// The frame rate.
    pub fn fps(&self) -> f64 {
        self.fps
    }

    /// The timing parameters in frames, `np.round(...).astype(int)`.
    ///
    /// numpy rounds halves to even, which is why `round_ties_even` is used here: at 100 fps
    /// a `pre_avg` of 0.025 s is 2.5 frames and must become 2, not 3.
    fn timings(&self) -> [usize; 5] {
        [
            self.smooth,
            self.pre_avg,
            self.post_avg,
            self.pre_max,
            self.post_max,
        ]
        .map(|seconds| (seconds * self.fps).round_ties_even() as usize)
    }

    /// Reset the online state.
    pub fn reset(&mut self) {
        self.buffer = None;
        self.counter = 0;
        self.last_onset = None;
    }

    /// Detect onsets in a whole activation function.
    pub fn process_offline(
        &self,
        activations: ArrayView1<'_, f64>,
    ) -> Result<Vec<f64>, OnsetError> {
        let timings = self.timings();
        let onsets = peak_picking(
            activations,
            self.threshold,
            Some(timings[0]),
            timings[1],
            timings[2],
            timings[3],
            timings[4],
        )?;
        let mut onsets: Vec<f64> = onsets
            .iter()
            .map(|frame| *frame as f64 / self.fps)
            .collect();
        if self.delay != 0.0 {
            for onset in onsets.iter_mut() {
                *onset += self.delay;
            }
        }
        if self.combine != 0.0 {
            onsets = combine_events(&onsets, self.combine, Combine::Left);
        }
        Ok(onsets)
    }

    /// Detect onsets frame by frame, using only past information.
    ///
    /// The buffer holds the activations the peak picker needs to look back at; the first
    /// call prepends `pre_max` frames of zeros and offsets the frame counter accordingly,
    /// exactly like the reference's `BufferProcessor`.
    pub fn process_online(
        &mut self,
        activations: &[f64],
        reset: bool,
    ) -> Result<Vec<f64>, OnsetError> {
        let counter = match self.buffer.as_mut() {
            // keep buffering into the existing buffer
            Some(buffer) if !reset => {
                let buffer_length = buffer.len();
                if activations.len() >= buffer_length {
                    let start = activations.len() - buffer_length;
                    buffer.copy_from_slice(&activations[start..]);
                } else {
                    buffer.rotate_left(activations.len());
                    let start = buffer_length - activations.len();
                    buffer[start..].copy_from_slice(activations);
                }
                self.counter
            }
            // first call (or an explicit reset): prepend `pre_max` frames of zeros and
            // offset the counter, exactly like the reference's BufferProcessor
            _ => {
                self.reset();
                let initial = vec![0.0; (self.pre_max * self.fps).round_ties_even() as usize];
                let mut buffer = initial.clone();
                buffer.extend_from_slice(activations);
                self.counter = -(initial.len() as i64);
                self.buffer = Some(buffer);
                self.counter
            }
        };
        let timings = self.timings();
        let buffer = self.buffer.clone().expect("the buffer exists");
        let peaks = peak_picking(
            ArrayView1::from(&buffer),
            self.threshold,
            Some(timings[0]),
            timings[1],
            timings[2],
            timings[3],
            timings[4],
        )?;
        let mut onsets: Vec<f64> = peaks
            .iter()
            .map(|peak| (counter + *peak as i64) as f64 / self.fps)
            .collect();
        self.counter += activations.len() as i64;
        if self.delay != 0.0 {
            return Err(OnsetError::DelayInOnlineMode);
        }
        if self.combine != 0.0 && !onsets.is_empty() {
            let mut start = 0usize;
            if let Some(last) = self.last_onset {
                let mut with_last = vec![last];
                with_last.extend(onsets.iter().copied());
                onsets = with_last;
                start = 1;
            }
            onsets = combine_events(&onsets, self.combine, Combine::Left);
            let last = onsets[onsets.len() - 1];
            if Some(last) != self.last_onset {
                self.last_onset = Some(last);
                onsets = onsets[start..].to_vec();
            } else {
                onsets.clear();
            }
        }
        Ok(onsets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn wrap_to_pi_keeps_the_range() {
        let values = array![[0.0, 3.0, -3.0, 6.5]];
        let wrapped = wrap_to_pi(values.view());
        for value in wrapped.iter() {
            assert!(*value > -std::f64::consts::PI - 1e-12);
            assert!(*value <= std::f64::consts::PI + 1e-12);
        }
        assert!((wrapped[[0, 1]] - 3.0).abs() < 1e-12);
        assert!((wrapped[[0, 2]] + 3.0).abs() < 1e-12);
    }

    #[test]
    fn peak_picking_finds_a_single_peak_above_the_threshold() {
        let activations = array![0.1, 0.2, 0.9, 0.2, 0.1];
        // no averaging, no maximum search: everything above the threshold is a detection
        let onsets = peak_picking(activations.view(), 0.5, None, 0, 0, 0, 0).unwrap();
        assert_eq!(onsets, vec![2]);
        // with a maximum search of one frame on both sides only the local maximum remains
        let onsets = peak_picking(activations.view(), 0.0, None, 0, 0, 1, 1).unwrap();
        assert_eq!(onsets, vec![2]);
    }

    #[test]
    fn peak_picking_keeps_only_local_maxima() {
        // a shoulder next to a higher peak is dropped once the maximum search is on:
        // frame 1 holds 0.5, but frame 2 within one frame holds 0.9
        let activations = array![0.0, 0.5, 0.9, 0.5, 0.0];
        let onsets = peak_picking(activations.view(), 0.4, None, 0, 0, 0, 0).unwrap();
        assert_eq!(onsets, vec![1, 2, 3]);
        let onsets = peak_picking(activations.view(), 0.4, None, 0, 0, 1, 1).unwrap();
        assert_eq!(onsets, vec![2], "only the local maximum survives");
    }

    #[test]
    fn moving_average_of_a_rise_detects_the_end_not_the_middle() {
        // `mode='constant'` counts the samples outside the activations as zero but still
        // divides by the window size, so the average of a rising slope *falls* at the end
        // and the last frames are the ones that stand out. This is the reference's
        // behaviour, not an artefact of the port.
        let activations = Array1::from((0..10).map(|value| value as f64 * 0.1).collect::<Vec<_>>());
        // both expectations are madmom's own outputs for exactly these arguments
        let onsets = peak_picking(activations.view(), 0.05, None, 2, 2, 0, 0).unwrap();
        assert_eq!(onsets, vec![8, 9]);
        let middle = activations.slice(ndarray::s![1..7]).to_owned();
        let onsets = peak_picking(middle.view(), 0.05, None, 2, 2, 0, 0).unwrap();
        assert_eq!(
            onsets,
            vec![4, 5],
            "the end of the slice behaves like the end of the signal"
        );
    }

    #[test]
    fn processor_converts_seconds_to_frames_with_half_to_even_rounding() {
        let processor =
            OnsetPeakPickingProcessor::new(0.5, 0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, false);
        // 0.025 s at 100 fps is 2.5 frames; numpy rounds it to 2
        assert_eq!(processor.timings()[0], 2);
        let processor = OnsetPeakPickingProcessor::default();
        assert_eq!(processor.timings(), [0, 0, 0, 0, 0]);
    }

    #[test]
    fn processor_combines_onsets_within_the_combine_window() {
        let processor =
            OnsetPeakPickingProcessor::new(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.0, 100.0, false);
        // peaks at frames 10, 12 (0.02 s apart) and 40
        let mut activations = vec![0.0; 50];
        activations[10] = 0.9;
        activations[12] = 0.9;
        activations[40] = 0.9;
        let onsets = processor
            .process_offline(ArrayView1::from(&activations))
            .unwrap();
        assert_eq!(onsets, vec![0.1, 0.4]);
    }

    #[test]
    fn online_mode_needs_no_future_information() {
        let mut processor =
            OnsetPeakPickingProcessor::new(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, true);
        let mut activations = [0.0; 20];
        activations[5] = 0.9;
        activations[15] = 0.9;
        let mut collected: Vec<f64> = Vec::new();
        for (frame, activation) in activations.iter().enumerate() {
            let onsets = processor.process_online(&[*activation], false).unwrap();
            for onset in onsets {
                assert!(
                    (onset - frame as f64 / 100.0).abs() < 1e-12,
                    "an online onset may not be in the future"
                );
                collected.push(onset);
            }
        }
        assert_eq!(collected.len(), 2, "both peaks reported: {collected:?}");
    }
}
