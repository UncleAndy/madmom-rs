//! Beat tracking with a dynamic Bayesian network approximated by an HMM.
//!
//! Port of the DBN part of `madmom/features/beats.py` (`DBNBeatTrackingProcessor`,
//! `threshold_activations`). The pipeline is the reference's: beat activations →
//! (optional thresholding) → Viterbi over the beat state space → one beat per region
//! where the observation model points at the beat column, taking the frame with the
//! highest activation inside that region → seconds.
//!
//! Defaults are the reference's: 55..215 bpm, `transition_lambda` 100,
//! `observation_lambda` 16, threshold 0, `correct` on.

use madmom_ml::hmm::{HiddenMarkovModel, ObservationModel, ViterbiResult};
use ndarray::Array2;

use crate::beats_hmm::{BeatStateSpace, BeatTransitionModel, RNNBeatTrackingObservationModel};

/// Default minimum tempo [bpm].
pub const MIN_BPM: f64 = 55.0;
/// Default maximum tempo [bpm].
pub const MAX_BPM: f64 = 215.0;
/// Default lambda of the exponential tempo transition.
pub const TRANSITION_LAMBDA: f64 = 100.0;
/// Default splitting of one beat period into beat and no-beat parts.
pub const OBSERVATION_LAMBDA: f64 = 16.0;
/// Default activation threshold (0 disables thresholding).
pub const THRESHOLD: f64 = 0.0;
/// Beats are aligned to the activation peak by default.
pub const CORRECT: bool = true;

/// Keep only the main segment of activations that reaches `threshold`.
///
/// Returns the segment and the index of its first frame, exactly like the reference:
/// thresholding is not applied frame by frame but as one contiguous segment from the
/// first to the last frame above the threshold.
pub fn threshold_activations(activations: &[f64], threshold: f64) -> (Vec<f64>, usize) {
    let first = activations.iter().position(|value| *value >= threshold);
    let last = activations.iter().rposition(|value| *value >= threshold);
    match (first, last) {
        (Some(first), Some(last)) => (activations[first..=last].to_vec(), first),
        _ => (Vec::new(), 0),
    }
}

/// Beat tracking with an HMM over a beat state space.
#[derive(Debug, Clone)]
pub struct DBNBeatTrackingProcessor {
    state_space: BeatStateSpace,
    transition_model: BeatTransitionModel,
    hmm: HiddenMarkovModel<RNNBeatTrackingObservationModel>,
    min_bpm: f64,
    max_bpm: f64,
    transition_lambda: f64,
    observation_lambda: f64,
    threshold: f64,
    correct: bool,
    fps: f64,
}

impl DBNBeatTrackingProcessor {
    /// Build a processor.
    ///
    /// `fps` is required: the reference derives the state space from the frame rate
    /// (`min_interval = 60 * fps / max_bpm`, `max_interval = 60 * fps / min_bpm`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        min_bpm: f64,
        max_bpm: f64,
        num_tempi: Option<usize>,
        transition_lambda: f64,
        observation_lambda: f64,
        threshold: f64,
        correct: bool,
        fps: f64,
    ) -> Self {
        let min_interval = 60.0 * fps / max_bpm;
        let max_interval = 60.0 * fps / min_bpm;
        let state_space = BeatStateSpace::new(min_interval, max_interval, num_tempi);
        let transition_model = BeatTransitionModel::new(state_space.clone(), transition_lambda);
        let observation_model =
            RNNBeatTrackingObservationModel::new(&state_space, observation_lambda);
        let hmm = HiddenMarkovModel::new(
            transition_model.clone().into_transition_model(),
            observation_model,
            None,
        )
        .expect("the beat HMM starts from a uniform distribution");
        Self {
            state_space,
            transition_model,
            hmm,
            min_bpm,
            max_bpm,
            transition_lambda,
            observation_lambda,
            threshold,
            correct,
            fps,
        }
    }

    /// Build a processor with the reference's defaults for a given frame rate.
    pub fn with_defaults(fps: f64) -> Self {
        Self::new(
            MIN_BPM,
            MAX_BPM,
            None,
            TRANSITION_LAMBDA,
            OBSERVATION_LAMBDA,
            THRESHOLD,
            CORRECT,
            fps,
        )
    }

    /// The beat state space.
    pub fn state_space(&self) -> &BeatStateSpace {
        &self.state_space
    }

    /// The transition model.
    pub fn transition_model(&self) -> &BeatTransitionModel {
        &self.transition_model
    }

    /// The HMM used for decoding.
    pub fn hmm(&self) -> &HiddenMarkovModel<RNNBeatTrackingObservationModel> {
        &self.hmm
    }

    /// The frame rate the processor works at.
    pub fn fps(&self) -> f64 {
        self.fps
    }

    /// Configured tempo limits.
    pub fn bpm_range(&self) -> (f64, f64) {
        (self.min_bpm, self.max_bpm)
    }

    /// Configured lambdas `(transition, observation)`.
    pub fn lambdas(&self) -> (f64, f64) {
        (self.transition_lambda, self.observation_lambda)
    }

    /// Decode the state path of the activations (no thresholding).
    ///
    /// The beat observation model takes one value per frame, so the activations are
    /// passed as a single-column matrix.
    pub fn viterbi_path(&self, activations: &[f64]) -> ViterbiResult {
        let matrix = Array2::from_shape_vec((activations.len(), 1), activations.to_vec())
            .expect("one column holds all activations");
        self.hmm.viterbi(matrix.view())
    }

    /// Detect beat positions [seconds] from beat activations.
    pub fn process_offline(&self, activations: &[f64]) -> Result<Vec<f64>, &'static str> {
        let mut beats: Vec<usize> = Vec::new();
        let mut first = 0usize;
        let (activations, thresholded_first) = if self.threshold != 0.0 {
            threshold_activations(activations, self.threshold)
        } else {
            (activations.to_vec(), 0)
        };
        first += thresholded_first;
        if activations.iter().all(|value| *value == 0.0) {
            return Ok(Vec::new());
        }
        let result = self.viterbi_path(&activations);
        if result.path.is_empty() {
            return Ok(Vec::new());
        }
        if self.correct {
            // regions where the observation model points at the beat column
            let pointers = self.hmm.observation_model().pointers();
            let beat_range: Vec<bool> = result
                .path
                .iter()
                .map(|state| pointers[*state as usize] == 1)
                .collect();
            let mut regions: Vec<(usize, usize)> = Vec::new();
            let mut start: Option<usize> = None;
            for (frame, in_beat) in beat_range.iter().enumerate() {
                match (in_beat, start) {
                    (true, None) => start = Some(frame),
                    (false, Some(from)) => {
                        regions.push((from, frame));
                        start = None;
                    }
                    _ => {}
                }
            }
            if let Some(from) = start {
                regions.push((from, beat_range.len()));
            }
            for (from, to) in regions {
                // the first frame with the highest activation wins, like np.argmax
                let mut best = from;
                for frame in from + 1..to {
                    if activations[frame] > activations[best] {
                        best = frame;
                    }
                }
                beats.push(best);
            }
        } else {
            return Err("correct=false needs scipy's argrelmin (plan task 3.x)");
        }
        Ok(beats
            .into_iter()
            .map(|beat| (beat + first) as f64 / self.fps)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thresholding_keeps_the_main_segment() {
        let activations = vec![0.0, 0.1, 0.5, 0.0, 0.6, 0.2, 0.0];
        // the segment runs from the first to the *last* frame above the threshold:
        // index 2 (0.5) to index 4 (0.6), so the 0.2 at index 5 stays out
        let (segment, first) = threshold_activations(&activations, 0.3);
        assert_eq!(first, 2);
        assert_eq!(segment, vec![0.5, 0.0, 0.6]);
    }

    #[test]
    fn state_space_follows_the_reference_interval_formula() {
        // fps 100, 55..215 bpm -> intervals round(27.9)..=round(109.1) = 28..=109
        let processor = DBNBeatTrackingProcessor::with_defaults(100.0);
        let state_space = processor.state_space();
        assert_eq!(state_space.intervals().first(), Some(&28));
        assert_eq!(state_space.intervals().last(), Some(&109));
        assert_eq!(state_space.num_intervals(), 82);
        // sum of 28..=109
        assert_eq!(state_space.num_states(), (28 + 109) * 82 / 2);
    }

    #[test]
    fn a_single_activation_yields_one_beat() {
        let processor = DBNBeatTrackingProcessor::with_defaults(100.0);
        let mut activations = vec![0.1f64; 100];
        activations[10] = 1.0;
        activations[60] = 1.0;
        let beats = processor.process_offline(&activations).unwrap();
        assert!(!beats.is_empty(), "expected at least one beat");
        for beat in &beats {
            assert!((0.0..=1.0).contains(beat), "beat {beat} out of range");
        }
    }
}
