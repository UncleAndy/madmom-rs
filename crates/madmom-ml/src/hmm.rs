//! Hidden Markov models and Viterbi decoding.
//!
//! Port of `madmom/ml/hmm.pyx` (753 lines of Cython, 9 tests in `tests/test_ml_hmm.py`).
//! The representation is the interesting part: transitions are stored **sparse** in a
//! CSR-like layout (`pointers[s]..pointers[s+1]` indexes the previous states of state
//! `s` and their log probabilities), which is what makes the beat DBN affordable —
//! its state space has thousands of states but only a handful of incoming transitions
//! each.
//!
//! Everything is computed in the **log domain** with the initial distribution
//! logarithmised, exactly as the reference does. Two consequences that are easy to get
//! wrong: an impossible sequence yields `-inf` and an **empty** path (the reference
//! warns and returns it, the caller has to check), and the backtracking pointers of the
//! first frame are never used.

use ndarray::{Array1, Array2, ArrayView2};
use thiserror::Error;

/// Errors of the HMM stage.
#[derive(Debug, Error, PartialEq)]
pub enum HmmError {
    /// The outgoing probabilities of some state do not sum to one.
    #[error("not a probability distribution: state {state} sums to {sum}")]
    NotAProbabilityDistribution {
        /// Previous state whose outgoing transitions do not sum to one.
        state: usize,
        /// Sum that was found.
        sum: f64,
    },
    /// The initial distribution does not sum to one.
    #[error("initial distribution does not sum to one: {0}")]
    InvalidInitialDistribution(f64),
    /// The transition model has no states at all.
    #[error("transition model has no states")]
    EmptyTransitionModel,
}

/// Sparse transition model, `states[pointers[s]..pointers[s+1]]` are the states that
/// transition into state `s`.
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionModel {
    states: Vec<u32>,
    pointers: Vec<u32>,
    probabilities: Vec<f64>,
    /// The logarithms of `probabilities`, computed once: the Viterbi loop reads them
    /// per transition and the reference likewise logs them once per decoding.
    log_probabilities: Vec<f64>,
}

impl TransitionModel {
    /// Build a transition model from its sparse representation.
    ///
    /// No validation beyond the lengths: this is the constructor the reference exposes
    /// for models it built itself.
    pub fn from_sparse(states: Vec<u32>, pointers: Vec<u32>, probabilities: Vec<f64>) -> Self {
        assert!(
            !pointers.is_empty(),
            "pointers must hold at least one entry (0 for an empty model)"
        );
        assert_eq!(
            states.len(),
            probabilities.len(),
            "each transition needs a probability"
        );
        let log_probabilities = probabilities.iter().map(|p| p.ln()).collect();
        Self {
            states,
            pointers,
            probabilities,
            log_probabilities,
        }
    }

    /// Build a transition model from dense transitions (the reference's `from_dense`).
    pub fn from_dense(
        states: &[u32],
        prev_states: &[u32],
        probabilities: &[f64],
    ) -> Result<Self, HmmError> {
        Self::make_sparse(states, prev_states, probabilities)
    }

    /// Build the sparse representation from a list of transitions, like `make_sparse`.
    ///
    /// `states[i]` is the destination state of transition `i`, `prev_states[i]` its
    /// origin, `probabilities[i]` its probability. The result is grouped by destination
    /// state, and the incoming states of each group are sorted ascending — the order
    /// scipy's CSR conversion produces, and the order Viterbi tie-breaking depends on.
    pub fn make_sparse(
        states: &[u32],
        prev_states: &[u32],
        probabilities: &[f64],
    ) -> Result<Self, HmmError> {
        assert_eq!(states.len(), prev_states.len());
        assert_eq!(states.len(), probabilities.len());
        if prev_states.is_empty() {
            return Err(HmmError::EmptyTransitionModel);
        }
        let num_states = prev_states.iter().copied().max().unwrap_or(0) as usize + 1;
        // every previous state must distribute its transitions like a distribution
        let mut outgoing = vec![0.0f64; num_states];
        for (index, prev) in prev_states.iter().enumerate() {
            outgoing[*prev as usize] += probabilities[index];
        }
        for (state, sum) in outgoing.iter().enumerate() {
            // np.allclose semantics: |a - b| <= atol + rtol * |b|
            if (sum - 1.0).abs() > 1e-8 + 1e-5 * 1.0 {
                return Err(HmmError::NotAProbabilityDistribution { state, sum: *sum });
            }
        }
        // group by destination state, sorted by the incoming state
        let mut transitions: Vec<(u32, u32, f64)> = states
            .iter()
            .zip(prev_states)
            .zip(probabilities)
            .map(|((state, prev), probability)| (*state, *prev, *probability))
            .collect();
        transitions.sort_by_key(|(state, prev, _)| (*state, *prev));
        let mut pointers = vec![0u32; num_states + 1];
        for (state, _, _) in &transitions {
            pointers[*state as usize + 1] += 1;
        }
        for index in 0..num_states {
            pointers[index + 1] += pointers[index];
        }
        let probabilities: Vec<f64> = transitions.iter().map(|(_, _, p)| *p).collect();
        let log_probabilities = probabilities.iter().map(|p| p.ln()).collect();
        Ok(Self {
            states: transitions.iter().map(|(_, prev, _)| *prev).collect(),
            pointers,
            probabilities,
            log_probabilities,
        })
    }

    /// Number of states.
    pub fn num_states(&self) -> usize {
        self.pointers.len() - 1
    }

    /// Number of transitions.
    pub fn num_transitions(&self) -> usize {
        self.probabilities.len()
    }

    /// Transition probabilities (not logarithms).
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    /// Transition log probabilities (all transitions, in storage order).
    pub fn log_probabilities(&self) -> &[f64] {
        &self.log_probabilities
    }

    /// Previous states of `state`, i.e. `states[pointers[state]..pointers[state + 1]]`.
    pub fn previous_states(&self, state: usize) -> &[u32] {
        let start = self.pointers[state] as usize;
        let end = self.pointers[state + 1] as usize;
        &self.states[start..end]
    }

    /// Transition probabilities of `state` (the forward algorithm reads these).
    pub fn probabilities_of(&self, state: usize) -> &[f64] {
        let start = self.pointers[state] as usize;
        let end = self.pointers[state + 1] as usize;
        &self.probabilities[start..end]
    }

    /// Transition log probabilities of `state`.
    pub fn log_probabilities_of(&self, state: usize) -> &[f64] {
        let start = self.pointers[state] as usize;
        let end = self.pointers[state + 1] as usize;
        &self.log_probabilities[start..end]
    }
}

/// Observation model: log densities plus the column each state observes.
pub trait ObservationModel {
    /// The column of the density matrix that each state observes.
    fn pointers(&self) -> &[u32];

    /// Log densities, `(num_observations, num_columns)`.
    ///
    /// Observations are 2-D: the beat models use a single column, the downbeat models
    /// two (beat and downbeat activations), exactly as the reference passes them.
    fn log_densities(&self, observations: ArrayView2<'_, f64>) -> Array2<f64>;

    /// The same densities in the linear domain.
    ///
    /// The forward algorithm normalises frame by frame in the linear domain (the
    /// reference's `forward` says so explicitly: it is faster than staying in logs), while
    /// Viterbi works in logs. `exp(log(p))` is exact enough here — the two agree wherever
    /// the density is not zero, and where it is zero both give zero.
    fn densities(&self, observations: ArrayView2<'_, f64>) -> Array2<f64> {
        self.log_densities(observations).mapv(|value| value.exp())
    }
}

/// A discrete observation model: a `(num_states, num_observations)` probability table.
///
/// The reference's `DiscreteObservationModel`: each row is a state's distribution over
/// the observation values, the rows must sum to one, and the pointers of the states are
/// simply their row indices.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteObservationModel {
    observation_probabilities: Array2<f64>,
    pointers: Vec<u32>,
}

impl DiscreteObservationModel {
    /// Build the model, validating that every state's probabilities sum to one.
    pub fn new(observation_probabilities: Array2<f64>) -> Result<Self, HmmError> {
        for (state, row) in observation_probabilities.rows().into_iter().enumerate() {
            let sum: f64 = row.sum();
            if (sum - 1.0).abs() > 1e-8 + 1e-5 {
                return Err(HmmError::NotAProbabilityDistribution { state, sum });
            }
        }
        let pointers = (0..observation_probabilities.shape()[0] as u32).collect();
        Ok(Self {
            observation_probabilities,
            pointers,
        })
    }
}

impl ObservationModel for DiscreteObservationModel {
    fn pointers(&self) -> &[u32] {
        &self.pointers
    }

    /// Densities of the observations, `observation_probabilities[:, observations].T`.
    fn log_densities(&self, observations: ArrayView2<'_, f64>) -> Array2<f64> {
        let num_states = self.observation_probabilities.shape()[0];
        let num_observations = self.observation_probabilities.shape()[1];
        let mut out = Array2::<f64>::zeros((observations.len(), num_states));
        for (frame, row) in observations.rows().into_iter().enumerate() {
            let observation = row[0] as usize;
            assert!(
                observation < num_observations,
                "observation {observation} is outside the model's {num_observations} values"
            );
            for state in 0..num_states {
                out[[frame, state]] = self.observation_probabilities[[state, observation]].ln();
            }
        }
        out
    }
}

/// Result of a Viterbi decoding.
#[derive(Debug, Clone, PartialEq)]
pub struct ViterbiResult {
    /// The best state path; empty when no valid path exists (`-inf` probability).
    pub path: Vec<u32>,
    /// Log probability of that path.
    pub log_probability: f64,
}

/// A hidden Markov model: transition model, observation model, initial distribution.
#[derive(Debug, Clone)]
pub struct HiddenMarkovModel<O: ObservationModel> {
    transition_model: TransitionModel,
    observation_model: O,
    initial_distribution: Array1<f64>,
    /// Forward variables of the previous frame, carried between calls in online mode.
    prev: Vec<f64>,
}

impl<O: ObservationModel> HiddenMarkovModel<O> {
    /// Build the model; without an initial distribution a uniform one is used.
    pub fn new(
        transition_model: TransitionModel,
        observation_model: O,
        initial_distribution: Option<Array1<f64>>,
    ) -> Result<Self, HmmError> {
        let initial_distribution = match initial_distribution {
            Some(distribution) => {
                let sum = distribution.sum();
                if (sum - 1.0).abs() > 1e-8 {
                    return Err(HmmError::InvalidInitialDistribution(sum));
                }
                distribution
            }
            None => {
                let num_states = transition_model.num_states();
                Array1::from_elem(num_states, 1.0 / num_states as f64)
            }
        };
        Ok(Self {
            transition_model,
            observation_model,
            prev: initial_distribution.to_vec(),
            initial_distribution,
        })
    }

    /// The transition model.
    pub fn transition_model(&self) -> &TransitionModel {
        &self.transition_model
    }

    /// The observation model.
    pub fn observation_model(&self) -> &O {
        &self.observation_model
    }

    /// The initial distribution.
    pub fn initial_distribution(&self) -> &Array1<f64> {
        &self.initial_distribution
    }

    /// Reset the forward variables to the initial distribution.
    pub fn reset(&mut self) {
        self.prev = self.initial_distribution.to_vec();
    }

    /// The forward variables, `(num_observations, num_states)`.
    ///
    /// Port of `HiddenMarkovModel.forward`: the recursion runs in the **linear** domain and
    /// every frame is normalised by the sum over the states, so the values stay in `0..1`
    /// and no logarithms are needed. With `reset = false` the recursion continues from the
    /// previous call, which is what the online beat tracker uses.
    ///
    /// The division is unguarded, like the reference's (`@cython.cdivision(True)`): a frame
    /// whose forward sum is zero produces infinities rather than an error.
    pub fn forward(&mut self, observations: ArrayView2<'_, f64>, reset: bool) -> Array2<f64> {
        let num_frames = observations.shape()[0];
        let num_states = self.transition_model.num_states();
        let densities = self.observation_model.densities(observations);
        if reset {
            self.reset();
        }
        let pointers = self.observation_model.pointers();
        let mut out = Array2::<f64>::zeros((num_frames, num_states));
        for frame in 0..num_frames {
            let mut sum = 0.0;
            for state in 0..num_states {
                let previous_states = self.transition_model.previous_states(state);
                let probabilities = self.transition_model.probabilities_of(state);
                let mut value = 0.0;
                for (previous, probability) in previous_states.iter().zip(probabilities.iter()) {
                    value += self.prev[*previous as usize] * *probability;
                }
                value *= densities[[frame, pointers[state] as usize]];
                out[[frame, state]] = value;
                sum += value;
            }
            let norm = 1.0 / sum;
            for state in 0..num_states {
                out[[frame, state]] *= norm;
                self.prev[state] = out[[frame, state]];
            }
        }
        out
    }

    /// Decode the best state path with the Viterbi algorithm.
    ///
    /// Mirrors `HiddenMarkovModel.viterbi`: log domain, sparse predecessor iteration,
    /// first maximum wins on ties, and a `-inf` result returns an empty path (the
    /// reference emits a warning there; the empty path is the contract callers rely on).
    pub fn viterbi(&self, observations: ArrayView2<'_, f64>) -> ViterbiResult {
        let num_states = self.transition_model.num_states();
        let num_observations = observations.shape()[0];
        let densities = self.observation_model.log_densities(observations);
        let pointers = self.observation_model.pointers();
        // ln(initial distribution) as the previous Viterbi variables
        let mut previous: Vec<f64> = self.initial_distribution.iter().map(|p| p.ln()).collect();
        let mut current = vec![0.0f64; num_states];
        let mut backtrack = vec![0u32; num_observations * num_states];

        for frame in 0..num_observations {
            for state in 0..num_states {
                let mut best = f64::NEG_INFINITY;
                let density = densities[[frame, pointers[state] as usize]];
                let previous_states = self.transition_model.previous_states(state);
                let log_probabilities = self.transition_model.log_probabilities_of(state);
                for (prev, log_probability) in previous_states.iter().zip(log_probabilities) {
                    let transition_probability =
                        previous[*prev as usize] + log_probability + density;
                    if transition_probability > best {
                        best = transition_probability;
                        backtrack[frame * num_states + state] = *prev;
                    }
                }
                current[state] = best;
            }
            previous.copy_from_slice(&current);
        }

        // best final state (first maximum, like np.argmax)
        let mut best_state = 0usize;
        let mut best_probability = f64::NEG_INFINITY;
        for (state, value) in current.iter().enumerate() {
            if *value > best_probability {
                best_probability = *value;
                best_state = state;
            }
        }
        if best_probability.is_infinite() {
            return ViterbiResult {
                path: Vec::new(),
                log_probability: best_probability,
            };
        }
        let mut path = vec![0u32; num_observations];
        let mut state = best_state;
        for frame in (0..num_observations).rev() {
            path[frame] = state as u32;
            state = backtrack[frame * num_states + state] as usize;
        }
        ViterbiResult {
            path,
            log_probability: best_probability,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Two states, staying where you are with 0.9, switching with 0.1.
    fn two_state_hmm() -> HiddenMarkovModel<DiscreteObservationModel> {
        let transitions =
            TransitionModel::make_sparse(&[0, 1, 0, 1], &[0, 0, 1, 1], &[0.9, 0.1, 0.1, 0.9])
                .expect("valid distribution");
        assert_eq!(transitions.num_states(), 2);
        assert_eq!(transitions.previous_states(0), &[0, 1]);
        assert_eq!(transitions.previous_states(1), &[0, 1]);
        // rows are states, columns are observation values
        let observations =
            DiscreteObservationModel::new(array![[0.9, 0.1], [0.2, 0.8]]).expect("distribution");
        HiddenMarkovModel::new(transitions, observations, None).expect("uniform prior")
    }

    #[test]
    fn viterbi_matches_the_reference_doctest() {
        // madmom's own HiddenMarkovModel doctest: 2 states, 3 observation values
        let transitions =
            TransitionModel::from_dense(&[0, 1, 0, 1], &[0, 0, 1, 1], &[0.7, 0.3, 0.6, 0.4])
                .expect("valid distribution");
        let observations = DiscreteObservationModel::new(array![[0.2, 0.3, 0.5], [0.7, 0.1, 0.2]])
            .expect("distribution");
        let hmm = HiddenMarkovModel::new(transitions, observations, None).expect("uniform prior");
        let result = hmm.viterbi(
            array![
                [0.0f64],
                [0.0],
                [1.0],
                [1.0],
                [0.0],
                [0.0],
                [0.0],
                [2.0],
                [2.0]
            ]
            .view(),
        );
        assert_eq!(result.path, vec![1, 1, 0, 0, 1, 1, 1, 0, 0]);
        // the doctest prints -12.87... (ellipsis, so the tail is not pinned there)
        assert!(
            (result.log_probability - (-12.87)).abs() < 0.005,
            "log probability {} does not match the reference's -12.87...",
            result.log_probability
        );
    }

    #[test]
    fn viterbi_follows_the_observations() {
        let hmm = two_state_hmm();
        // observations 0, 0, 1, 1 select state 0 and then state 1
        let result = hmm.viterbi(array![[0.0f64], [0.0], [1.0], [1.0]].view());
        assert_eq!(result.path, vec![0, 0, 1, 1]);
        assert!(result.log_probability.is_finite());
    }

    #[test]
    fn viterbi_reports_the_log_probability_of_the_path() {
        let hmm = two_state_hmm();
        let result = hmm.viterbi(array![[0.0f64]].view());
        // ln(1/2) + ln(0.9) + ln(0.9) for state 0 (transition from the prior)
        let expected = (0.5f64).ln() + (0.9f64).ln() + (0.9f64).ln();
        assert!((result.log_probability - expected).abs() < 1e-9);
    }

    #[test]
    fn impossible_observations_give_an_empty_path() {
        let transitions = TransitionModel::make_sparse(&[0], &[0], &[1.0]).unwrap();
        // density 0 -> log density -inf -> no valid path
        let observations = DiscreteObservationModel::new(array![[1.0f64, 0.0]]).unwrap();
        let hmm = HiddenMarkovModel::new(transitions, observations, None).unwrap();
        let result = hmm.viterbi(array![[1.0f64]].view());
        assert!(result.path.is_empty());
        assert!(result.log_probability.is_infinite());
    }

    #[test]
    fn a_transition_model_that_is_not_a_distribution_is_refused() {
        let error = TransitionModel::make_sparse(&[0, 1], &[0, 0], &[0.5, 0.4]).unwrap_err();
        assert!(matches!(
            error,
            HmmError::NotAProbabilityDistribution { .. }
        ));
    }

    #[test]
    fn sparse_layout_is_grouped_by_destination_and_sorted() {
        // transitions written in a jumbled order must come out grouped and sorted
        let model =
            TransitionModel::make_sparse(&[1, 0, 1, 0], &[1, 1, 0, 0], &[0.9, 0.1, 0.1, 0.9])
                .unwrap();
        assert_eq!(model.previous_states(0), &[0, 1]);
        assert_eq!(model.previous_states(1), &[0, 1]);
        assert_eq!(model.pointers, vec![0, 2, 4]);
    }
}
