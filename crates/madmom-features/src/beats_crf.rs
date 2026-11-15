//! CRF beat detection (`madmom/features/beats_crf.py`, Cython).
//!
//! A conditional random field over beat positions: instead of an HMM whose transitions
//! encode a tempo distribution over intervals, the CRF's transition distribution is a
//! log-normal distribution over **interval changes** (`norm.pdf(log2(move), log2(interval),
//! interval_sigma)`), and the per-frame normalisation factors come from correlating the
//! activations with that distribution.
//!
//! The reference runs the whole thing in `float32` (the Cython signatures declare
//! `float[::1]`), with `double` accumulators in the recursion, and computes the logarithms
//! of the distributions and the activations up front — so zeros become `-inf`, which is
//! what the Viterbi recursion expects.
//!
//! Everything here is a direct port; the only judgement call is that the arrays are `f32`
//! and the accumulators `f64`, exactly as the Cython declares them.

use ndarray::{Array1, ArrayView1};

use madmom_audio::signal::smooth_f64;
use madmom_core::ndimage::correlate1d_f32;

use crate::tempo::{
    CombFilterTempoHistogramProcessor, TempoError, TempoEstimationProcessor, TempoHistogramSource,
};

/// Allowed deviation from the dominant interval (the reference's `INTERVAL_SIGMA`).
pub const INTERVAL_SIGMA: f64 = 0.18;
/// Factors of the dominant interval tried when `use_factors` is set.
pub const FACTORS: [f64; 5] = [0.5, 0.67, 1.0, 1.5, 2.0];
/// Number of estimated intervals tried by default.
pub const NUM_INTERVALS: usize = 5;
/// The CRF's own tempo range, used by the command line programs.
pub const MIN_BPM: f64 = 20.0;
/// The CRF's own maximum tempo.
pub const MAX_BPM: f64 = 240.0;
/// The CRF's own activation smoothing [seconds].
pub const ACT_SMOOTH: f64 = 0.09;
/// The CRF's own histogram smoothing [bins].
pub const HIST_SMOOTH: usize = 7;

/// The CRF's initial distribution: uniform over the first beat interval.
pub fn initial_distribution(num_states: usize, interval: usize) -> Array1<f32> {
    let mut distribution = Array1::<f32>::ones(num_states);
    if interval < num_states {
        for state in interval..num_states {
            distribution[state] = 0.0;
        }
    }
    distribution
}

/// The standard normal probability density, as `scipy.stats.norm.pdf`.
fn normal_pdf(value: f64, mean: f64, sigma: f64) -> f64 {
    let z = (value - mean) / sigma;
    (-0.5 * z * z).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt())
}

/// The transition distribution over interval changes.
///
/// The first entry of the move range is set to `1e-6` before taking the logarithm (the
/// reference does this so that `log2(0)` cannot appear), and the distribution is normalised
/// before being cast to `float32`.
pub fn transition_distribution(interval: usize, interval_sigma: f64) -> Array1<f32> {
    let mut move_range: Vec<f64> = (0..interval * 2).map(|move_| move_ as f64).collect();
    if !move_range.is_empty() {
        move_range[0] = 0.000001;
    }
    let mean = (interval as f64).log2();
    let mut distribution: Vec<f64> = move_range
        .iter()
        .map(|move_| normal_pdf(move_.log2(), mean, interval_sigma))
        .collect();
    let sum: f64 = distribution.iter().sum();
    for value in distribution.iter_mut() {
        *value /= sum;
    }
    Array1::from(
        distribution
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>(),
    )
}

/// The normalisation factors: the activations correlated with the transition distribution.
///
/// `correlate1d(..., origin=-int(len / 2))` makes the window reach **forward** from each
/// frame, which is what a CRF over beat positions needs: the factor at frame `i` describes
/// the probability of the interval that starts there.
pub fn normalisation_factors(activations: &[f32], transition: &[f32]) -> Array1<f32> {
    correlate1d_f32(activations, transition, 0.0)
}

/// Extract the most likely beat sequence for a piece with the Viterbi algorithm.
pub fn best_sequence(
    activations: &[f32],
    interval: usize,
    interval_sigma: f64,
) -> (Vec<usize>, f64) {
    let init = initial_distribution(activations.len(), interval);
    let transition = transition_distribution(interval, interval_sigma);
    let normalisation = normalisation_factors(activations, transition.as_slice().unwrap());
    // the reference logs every array up front (`with np.errstate(divide='ignore')`), so a
    // zero becomes -inf and the recursion treats it as impossible
    let init: Vec<f32> = init.iter().map(|value| value.ln()).collect();
    let transition: Vec<f32> = transition.iter().map(|value| value.ln()).collect();
    let normalisation: Vec<f32> = normalisation.iter().map(|value| value.ln()).collect();
    let activations: Vec<f32> = activations.iter().map(|value| value.ln()).collect();
    viterbi(&init, &transition, &normalisation, &activations, interval)
}

/// The Viterbi recursion of the CRF.
///
/// `pi`, `transition`, `norm_factor` and `activations` are already logarithms. The
/// recursion runs over the `num_states / tau` beats the piece can hold, and the back
/// pointers are stored per step — the reference keeps them in a `(num_x - 1, num_states)`
/// array, which is why the port does too.
pub fn viterbi(
    pi: &[f32],
    transition: &[f32],
    norm_factor: &[f32],
    activations: &[f32],
    tau: usize,
) -> (Vec<usize>, f64) {
    let num_states = activations.len();
    let num_transitions = transition.len();
    let num_beats = num_states / tau;
    if num_beats == 0 {
        return (Vec::new(), f64::NEG_INFINITY);
    }
    let mut previous: Vec<f64> = (0..num_states)
        .map(|state| pi[state] as f64 + activations[state] as f64 + norm_factor[state] as f64)
        .collect();
    let mut current = vec![f64::NEG_INFINITY; num_states];
    let mut back_pointers = vec![0usize; (num_beats - 1) * num_states];
    for step in 0..num_beats - 1 {
        current.fill(f64::NEG_INFINITY);
        for state in 0..num_states {
            let limit = state.min(num_transitions);
            for move_ in 0..limit {
                let probability = previous[state - move_] + transition[move_] as f64;
                if probability > current[state] {
                    current[state] = probability;
                    back_pointers[step * num_states + state] = state - move_;
                }
            }
            current[state] += activations[state] as f64 + norm_factor[state] as f64;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    let mut path_probability = f64::NEG_INFINITY;
    let mut next_state = 0usize;
    for state in 0..num_states {
        previous[state] -= norm_factor[state] as f64;
        if previous[state] > path_probability {
            path_probability = previous[state];
            next_state = state;
        }
    }
    let mut path = vec![0usize; num_beats];
    path[num_beats - 1] = next_state;
    for step in (0..num_beats - 1).rev() {
        next_state = back_pointers[step * num_states + next_state];
        path[step] = next_state;
    }
    (path, path_probability)
}

/// CRF beat detection over several candidate intervals.
///
/// The tempo is estimated with the comb-filter histogram (the tempo module's defaults, not
/// the CRF's class constants — those are only used by the command line programs), the best
/// `num_intervals` intervals are tried, and the sequence whose log probability per beat is
/// highest wins.
#[derive(Debug, Clone)]
pub struct CRFBeatDetectionProcessor {
    interval_sigma: f64,
    use_factors: bool,
    num_intervals: usize,
    factors: Vec<f64>,
    fps: f64,
    estimator: TempoEstimationProcessor<CombFilterTempoHistogramProcessor>,
}

impl CRFBeatDetectionProcessor {
    /// Build a processor.
    pub fn new(
        interval_sigma: f64,
        use_factors: bool,
        num_intervals: usize,
        factors: Vec<f64>,
        fps: f64,
    ) -> Self {
        Self {
            interval_sigma,
            use_factors,
            num_intervals,
            factors,
            fps,
            estimator: TempoEstimationProcessor::with_defaults(
                CombFilterTempoHistogramProcessor::with_defaults(fps),
                fps,
            ),
        }
    }

    /// Build a processor with the reference's defaults.
    pub fn with_defaults(fps: f64) -> Self {
        Self::new(INTERVAL_SIGMA, false, NUM_INTERVALS, FACTORS.to_vec(), fps)
    }

    /// The interval sigma in use.
    pub fn interval_sigma(&self) -> f64 {
        self.interval_sigma
    }

    /// The candidate beat intervals, largest first (the reference's order).
    pub fn possible_intervals(&self, activations: &[f64]) -> Result<Vec<usize>, TempoError> {
        let tempi = self.estimator.process(ArrayView1::from(activations))?;
        let mut intervals: Vec<f64> = (0..tempi.shape()[0])
            .map(|row| self.fps * 60.0 / tempi[[row, 0]])
            .collect();
        if self.use_factors {
            let dominant = intervals.first().copied().unwrap_or(0.0);
            intervals = self
                .factors
                .iter()
                .map(|factor| dominant * factor)
                .collect();
            let min = self.estimator.histogram_processor().min_interval() as f64;
            let max = self.estimator.histogram_processor().max_interval() as f64;
            intervals.retain(|interval| *interval >= min && *interval <= max);
        } else {
            intervals.truncate(self.num_intervals);
        }
        intervals
            .sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
        Ok(intervals
            .into_iter()
            .rev()
            .map(|interval| interval as usize)
            .collect())
    }

    /// Detect beat positions [seconds].
    pub fn process(&self, activations: &[f64]) -> Result<Vec<f64>, TempoError> {
        let intervals = self.possible_intervals(activations)?;
        if intervals.is_empty() {
            return Err(TempoError::NoBeats);
        }
        // smooth the activations the way the tempo estimator would
        let act_smooth = (self.fps * self.estimator.act_smooth().unwrap_or(0.0)) as usize;
        let smoothed = smooth_f64(ArrayView1::from(activations), act_smooth)?;
        // the reference makes the activations C-contiguous float32 for the Cython code
        let activations: Vec<f32> = smoothed.iter().map(|value| *value as f32).collect();
        let results: Vec<(Vec<usize>, f64)> = intervals
            .iter()
            .map(|interval| best_sequence(&activations, *interval, self.interval_sigma))
            .collect();
        // the sequence with the highest log probability per beat wins
        let mut best = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (index, (path, probability)) in results.iter().enumerate() {
            let score = probability / path.len() as f64;
            if score > best_score {
                best_score = score;
                best = index;
            }
        }
        Ok(results[best]
            .0
            .iter()
            .map(|frame| *frame as f64 / self.fps)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_distribution_covers_the_first_interval() {
        let distribution = initial_distribution(10, 4);
        assert_eq!(distribution.len(), 10);
        assert_eq!(
            &distribution.as_slice().unwrap()[..4],
            &[1.0, 1.0, 1.0, 1.0]
        );
        assert!(distribution.as_slice().unwrap()[4..]
            .iter()
            .all(|value| *value == 0.0));
    }

    #[test]
    fn transition_distribution_is_a_log_normal_over_interval_changes() {
        let distribution = transition_distribution(10, 0.18);
        assert_eq!(distribution.len(), 20);
        let sum: f64 = distribution.iter().map(|value| *value as f64).sum();
        assert!((sum - 1.0).abs() < 1e-6, "normalised, got {sum}");
        // the distribution peaks at the interval itself (move = interval)
        let peak = distribution
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.partial_cmp(right.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, 10);
    }

    #[test]
    fn viterbi_returns_one_state_per_beat() {
        // a piece of 20 frames with a beat every 5 frames holds 4 beats. Every activation
        // must stay positive: a zero becomes -inf, and with everything impossible the
        // recursion marches to state 0 (which is what the reference does as well — its
        // golden behaviour is pinned by tests/beats_trackers_golden.rs).
        let num_states = 20;
        let interval = 5;
        let mut activations = vec![0.01f32; num_states];
        for frame in (0..num_states).step_by(interval) {
            activations[frame] = 1.0;
        }
        let init: Vec<f32> = initial_distribution(num_states, interval)
            .iter()
            .map(|value| value.ln())
            .collect();
        let transition = transition_distribution(interval, INTERVAL_SIGMA);
        let transition_linear: Vec<f32> = transition.to_vec();
        let transition: Vec<f32> = transition.iter().map(|value| value.ln()).collect();
        let normalisation = normalisation_factors(&activations, &transition_linear);
        let normalisation: Vec<f32> = normalisation.iter().map(|value| value.ln()).collect();
        let log_activations: Vec<f32> = activations.iter().map(|value| value.ln()).collect();
        let (path, probability) = viterbi(
            &init,
            &transition,
            &normalisation,
            &log_activations,
            interval,
        );
        assert_eq!(path.len(), 4, "one state per beat");
        assert!(probability.is_finite(), "probability {probability}");
        assert!(
            path.windows(2).all(|pair| pair[1] > pair[0]),
            "the path must move forward: {path:?}"
        );
        // the exact positions are pinned by the golden test against madmom; here it is
        // enough that the recursion produces a forward moving, finite path
    }
}
