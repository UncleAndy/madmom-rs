//! State spaces and models of the beat-tracking dynamic Bayesian network.
//!
//! Port of the beat part of `madmom/features/beats_hmm.py` (721 lines, 17 tests in
//! `tests/test_features_beats_hmm.py`). The model is the efficient one from Krebs et
//! al. (ISMIR 2015): instead of one state per (interval, position) pair, the interval
//! itself is the state — consecutive frames within one interval are chained by
//! probability 1, and the tempo may only change when a beat is completed.
//!
//! Details that decide the numbers:
//!
//! * intervals are `round(min_interval)..=round(max_interval)`, or a deduplicated log
//!   spacing when `num_intervals` asks for fewer intervals (found iteratively, because
//!   the rounding decides how many distinct intervals survive);
//! * positions inside an interval are `linspace(0, 1, interval, endpoint=False)`;
//! * tempo transitions follow `exp(-lambda * |to/from - 1|)`, values below
//!   `np.spacing(1)` become exactly 0, and rows are normalized afterwards;
//! * the observation model maps the first `1/observation_lambda` of each beat period to
//!   the beat column (1) and the rest to the no-beat column (0).

use madmom_ml::hmm::{ObservationModel, TransitionModel};
use ndarray::{Array1, Array2, ArrayView1};

/// Exponent of the threshold below which tempo transitions are dropped
/// (`np.spacing(1)`, the smallest representable step above 1).
pub const TRANSITION_THRESHOLD: f64 = f64::EPSILON;

/// State space for beat tracking: one state per frame of each modelled interval.
#[derive(Debug, Clone)]
pub struct BeatStateSpace {
    intervals: Vec<usize>,
    num_states: usize,
    first_states: Vec<usize>,
    last_states: Vec<usize>,
    state_positions: Array1<f64>,
    state_intervals: Vec<usize>,
}

impl BeatStateSpace {
    /// Build the state space for intervals between `min_interval` and `max_interval`
    /// (both in frames).
    ///
    /// With `num_intervals` set (and smaller than the linear spacing) the intervals are
    /// log-spaced and deduplicated, exactly like the reference's iterative search.
    pub fn new(min_interval: f64, max_interval: f64, num_intervals: Option<usize>) -> Self {
        let mut intervals: Vec<usize> =
            (min_interval.round() as usize..=max_interval.round() as usize).collect();
        if let Some(requested) = num_intervals {
            if requested < intervals.len() {
                let mut count = requested;
                loop {
                    intervals = logspace_dedup(min_interval, max_interval, count);
                    if intervals.len() >= requested {
                        break;
                    }
                    count += 1;
                }
                intervals.truncate(requested.max(1));
            }
        }
        let num_states: usize = intervals.iter().sum();
        let mut first_states = Vec::with_capacity(intervals.len());
        let mut last_states = Vec::with_capacity(intervals.len());
        let mut running = 0usize;
        for interval in &intervals {
            first_states.push(running);
            last_states.push(running + interval - 1);
            running += *interval;
        }
        let mut state_positions = Array1::<f64>::zeros(num_states);
        let mut state_intervals = Vec::with_capacity(num_states);
        let mut index = 0usize;
        for interval in &intervals {
            for position in 0..*interval {
                state_positions[index + position] = position as f64 / *interval as f64;
                state_intervals.push(*interval);
            }
            index += *interval;
        }
        Self {
            intervals,
            num_states,
            first_states,
            last_states,
            state_positions,
            state_intervals,
        }
    }

    /// Number of states.
    pub fn num_states(&self) -> usize {
        self.num_states
    }

    /// Modelled intervals in frames.
    pub fn intervals(&self) -> &[usize] {
        &self.intervals
    }

    /// Number of modelled intervals.
    pub fn num_intervals(&self) -> usize {
        self.intervals.len()
    }

    /// First state of each interval.
    pub fn first_states(&self) -> &[usize] {
        &self.first_states
    }

    /// Last state of each interval.
    pub fn last_states(&self) -> &[usize] {
        &self.last_states
    }

    /// Position of every state inside its interval, `0..1`.
    pub fn state_positions(&self) -> ArrayView1<'_, f64> {
        self.state_positions.view()
    }

    /// Interval (in frames) that each state belongs to.
    pub fn state_intervals(&self) -> &[usize] {
        &self.state_intervals
    }
}

/// `np.logspace(log2(min), log2(max), count, base=2)`, rounded and deduplicated.
fn logspace_dedup(min_interval: f64, max_interval: f64, count: usize) -> Vec<usize> {
    let (start, stop) = (min_interval.log2(), max_interval.log2());
    let mut values: Vec<usize> = Vec::with_capacity(count);
    for index in 0..count {
        let fraction = if count == 1 {
            0.0
        } else {
            index as f64 / (count - 1) as f64
        };
        let value = 2f64.powf(start + (stop - start) * fraction);
        values.push(value.round() as usize);
    }
    values.sort_unstable();
    values.dedup();
    values
}

/// Exponential tempo transition probabilities between intervals.
///
/// `None` as lambda gives a diagonal matrix (only same-tempo transitions); otherwise
/// `exp(-lambda * |to/from - 1|)`, with values at or below [`TRANSITION_THRESHOLD`]
/// dropped to zero and the rows normalized.
pub fn exponential_transition(
    from_intervals: &[usize],
    to_intervals: &[usize],
    transition_lambda: Option<f64>,
) -> Array2<f64> {
    let (rows, columns) = (from_intervals.len(), to_intervals.len());
    if transition_lambda.is_none() {
        let mut matrix = Array2::<f64>::zeros((rows, columns));
        for index in 0..rows.min(columns) {
            matrix[[index, index]] = 1.0;
        }
        return matrix;
    }
    let lambda = transition_lambda.unwrap();
    let mut matrix = Array2::<f64>::zeros((rows, columns));
    for row in 0..rows {
        let from = from_intervals[row] as f64;
        for column in 0..columns {
            let ratio = to_intervals[column] as f64 / from;
            let probability = (-lambda * (ratio - 1.0).abs()).exp();
            matrix[[row, column]] = if probability <= TRANSITION_THRESHOLD {
                0.0
            } else {
                probability
            };
        }
        let sum: f64 = matrix.row(row).sum();
        if sum != 0.0 {
            for column in 0..columns {
                matrix[[row, column]] /= sum;
            }
        }
    }
    matrix
}

/// Transition model of the beat DBN.
#[derive(Debug, Clone)]
pub struct BeatTransitionModel {
    state_space: BeatStateSpace,
    transition_lambda: f64,
    transition_model: TransitionModel,
}

impl BeatTransitionModel {
    /// Build the transition model for a state space.
    pub fn new(state_space: BeatStateSpace, transition_lambda: f64) -> Self {
        // staying in the same interval has probability 1, and every state except the
        // first states of the intervals is reached from its predecessor
        let first_states = state_space.first_states().to_vec();
        let mut states: Vec<u32> = (0..state_space.num_states() as u32)
            .filter(|state| !first_states.contains(&(*state as usize)))
            .collect();
        let mut prev_states: Vec<u32> = states.iter().map(|state| state - 1).collect();
        let mut probabilities: Vec<f64> = vec![1.0; states.len()];
        // tempo changes happen between the last and the first state of the intervals
        let from_intervals: Vec<usize> = state_space
            .last_states()
            .iter()
            .map(|state| state_space.state_intervals()[*state])
            .collect();
        let to_intervals: Vec<usize> = state_space
            .first_states()
            .iter()
            .map(|state| state_space.state_intervals()[*state])
            .collect();
        let prob = exponential_transition(&from_intervals, &to_intervals, Some(transition_lambda));
        let mut extra_states = Vec::new();
        let mut extra_prev = Vec::new();
        let mut extra_probabilities = Vec::new();
        for row in 0..prob.shape()[0] {
            for column in 0..prob.shape()[1] {
                let probability = prob[[row, column]];
                if probability != 0.0 {
                    extra_states.push(state_space.first_states()[column] as u32);
                    extra_prev.push(state_space.last_states()[row] as u32);
                    extra_probabilities.push(probability);
                }
            }
        }
        states.extend(extra_states);
        prev_states.extend(extra_prev);
        probabilities.extend(extra_probabilities);
        let transition_model = TransitionModel::make_sparse(&states, &prev_states, &probabilities)
            .expect("the beat transition model must be a distribution");
        Self {
            state_space,
            transition_lambda,
            transition_model,
        }
    }

    /// The state space.
    pub fn state_space(&self) -> &BeatStateSpace {
        &self.state_space
    }

    /// The tempo transition lambda.
    pub fn transition_lambda(&self) -> f64 {
        self.transition_lambda
    }

    /// The underlying sparse transition model.
    pub fn transition_model(&self) -> &TransitionModel {
        &self.transition_model
    }

    /// The transition model itself (it is passed to the HMM by value).
    pub fn into_transition_model(self) -> TransitionModel {
        self.transition_model
    }
}

/// Observation model of the beat DBN: beat activations, split into beat and no-beat.
#[derive(Debug, Clone)]
pub struct RNNBeatTrackingObservationModel {
    observation_lambda: f64,
    pointers: Vec<u32>,
}

impl RNNBeatTrackingObservationModel {
    /// Build the observation model for a state space.
    ///
    /// States whose position is below `1 / observation_lambda` observe the beat column.
    pub fn new(state_space: &BeatStateSpace, observation_lambda: f64) -> Self {
        let border = 1.0 / observation_lambda;
        let pointers = state_space
            .state_positions()
            .iter()
            .map(|position| if *position < border { 1 } else { 0 })
            .collect();
        Self {
            observation_lambda,
            pointers,
        }
    }

    /// The observation lambda.
    pub fn observation_lambda(&self) -> f64 {
        self.observation_lambda
    }
}

impl ObservationModel for RNNBeatTrackingObservationModel {
    fn pointers(&self) -> &[u32] {
        &self.pointers
    }

    fn log_densities(&self, observations: ArrayView1<'_, f64>) -> Array2<f64> {
        let mut densities = Array2::<f64>::zeros((observations.len(), 2));
        let lambda = self.observation_lambda;
        for (frame, observation) in observations.iter().enumerate() {
            densities[[frame, 0]] = ((1.0 - observation) / (lambda - 1.0)).ln();
            densities[[frame, 1]] = observation.ln();
        }
        densities
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_space_counts_follow_the_reference_definition() {
        // intervals 2..=4 -> 2 + 3 + 4 = 9 states
        let state_space = BeatStateSpace::new(2.0, 4.0, None);
        assert_eq!(state_space.num_states(), 9);
        assert_eq!(state_space.intervals(), &[2, 3, 4]);
        assert_eq!(state_space.first_states(), &[0, 2, 5]);
        assert_eq!(state_space.last_states(), &[1, 4, 8]);
        // positions restart in every interval, linspace(0, 1, n, endpoint=False)
        assert_eq!(
            state_space.state_positions().to_vec(),
            vec![0.0, 0.5, 0.0, 1.0 / 3.0, 2.0 / 3.0, 0.0, 0.25, 0.5, 0.75]
        );
        assert_eq!(state_space.state_intervals(), &[2, 2, 3, 3, 3, 4, 4, 4, 4]);
    }

    #[test]
    fn log_spacing_is_deduplicated_and_limited() {
        // 3 intervals between 2 and 100 frames: the log spacing collapses duplicates,
        // so the reference keeps increasing the count until at least 3 survive
        let state_space = BeatStateSpace::new(2.0, 100.0, Some(3));
        assert_eq!(state_space.num_intervals(), 3);
        assert!(state_space.intervals()[0] >= 2);
        assert!(*state_space.intervals().last().unwrap() <= 100);
    }

    #[test]
    fn exponential_transition_normalizes_rows_and_drops_tiny_values() {
        let matrix = exponential_transition(&[10, 20], &[10, 20], Some(100.0));
        assert_eq!(matrix.shape(), &[2, 2]);
        for row in 0..2 {
            let sum: f64 = matrix.row(row).sum();
            assert!((sum - 1.0).abs() < 1e-12, "row {row} sums to {sum}");
        }
        // a tempo change by a factor of two is practically impossible with lambda = 100
        assert!(matrix[[0, 1]] < 1e-10);
    }

    #[test]
    fn no_lambda_gives_a_diagonal_matrix() {
        let matrix = exponential_transition(&[10, 20], &[10, 20], None);
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[1, 1]], 1.0);
        assert_eq!(matrix[[0, 1]], 0.0);
    }

    #[test]
    fn observation_pointers_split_the_beat_period() {
        let state_space = BeatStateSpace::new(4.0, 4.0, None);
        // lambda 4 -> positions < 0.25 are beat states, i.e. only position 0
        let model = RNNBeatTrackingObservationModel::new(&state_space, 4.0);
        assert_eq!(model.pointers(), &[1, 0, 0, 0]);
        let densities = model.log_densities(ndarray::array![0.5f64].view());
        assert!((densities[[0, 1]] - 0.5f64.ln()).abs() < 1e-12);
        assert!((densities[[0, 0]] - ((1.0 - 0.5) / 3.0f64).ln()).abs() < 1e-12);
    }
}
