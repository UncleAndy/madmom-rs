//! Downbeat tracking with the joint beat/downbeat DBN.
//!
//! Port of `DBNDownBeatTrackingProcessor` from `madmom/features/downbeats.py`. Several
//! bar lengths are modelled at once (one HMM per `beats_per_bar`), all of them decode the
//! same activations, and the one with the highest path log probability wins.
//!
//! Details that decide the numbers:
//!
//! * the defaults differ from the beat tracker: `NUM_TEMPI = 60` (a limited log spacing)
//!   and `THRESHOLD = 0.05` (the beat tracker has `None` and `0`);
//! * `threshold_activations` works on rows: a frame is kept when *any* column reaches the
//!   threshold;
//! * the peak correction searches the **flattened** two-column activation window and
//!   divides the index by two (`np.argmax(...) // 2 + left`), i.e. a beat and a downbeat
//!   value are compared as if they lived in one array;
//! * beat numbers come from `positions.astype(int) + 1`, so they start at 1.

use madmom_ml::hmm::{HiddenMarkovModel, ObservationModel, ViterbiResult};
use ndarray::{Array2, ArrayView2};

use crate::beats::{MAX_BPM, MIN_BPM};
use crate::beats_hmm::{BarStateSpace, BarTransitionModel, RNNDownBeatTrackingObservationModel};

/// Default number of modelled tempi for downbeats.
pub const NUM_TEMPI: usize = 60;
/// Default lambda of the exponential tempo transition.
pub const TRANSITION_LAMBDA: f64 = 100.0;
/// Default splitting of one beat period into beat and no-beat parts.
pub const OBSERVATION_LAMBDA: f64 = 16.0;
/// Default activation threshold (the beat tracker uses 0 instead).
pub const THRESHOLD: f64 = 0.05;
/// Beats are aligned to the activation peak by default.
pub const CORRECT: bool = true;

/// One detected beat: position in seconds and its number inside the bar (1-based).
#[derive(Debug, Clone, PartialEq)]
pub struct DownBeat {
    /// Position in seconds.
    pub time: f64,
    /// Beat number inside the bar, starting at 1.
    pub beat_number: i64,
}

/// Downbeat tracking over several bar lengths.
#[derive(Debug, Clone)]
pub struct DBNDownBeatTrackingProcessor {
    hmms: Vec<HiddenMarkovModel<RNNDownBeatTrackingObservationModel>>,
    state_spaces: Vec<BarStateSpace>,
    beats_per_bar: Vec<usize>,
    transition_lambda: Vec<Option<f64>>,
    observation_lambda: f64,
    threshold: f64,
    correct: bool,
    fps: f64,
}

impl DBNDownBeatTrackingProcessor {
    /// Build a processor for the given bar lengths.
    ///
    /// A single `min_bpm`/`max_bpm`/`num_tempi`/`transition_lambda` value is used for
    /// every bar length, like the reference's `np.repeat`.
    // The argument list mirrors the reference's constructor one for one; grouping it into
    // a struct would make the port harder to check against `downbeats.py`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        beats_per_bar: &[usize],
        min_bpm: &[f64],
        max_bpm: &[f64],
        num_tempi: &[usize],
        transition_lambda: &[Option<f64>],
        observation_lambda: f64,
        threshold: f64,
        correct: bool,
        fps: f64,
    ) -> Self {
        let lengths = beats_per_bar.len();
        let min_bpm = repeat_to(min_bpm, lengths);
        let max_bpm = repeat_to(max_bpm, lengths);
        let num_tempi = repeat_to(num_tempi, lengths);
        let transition_lambda = repeat_to(transition_lambda, lengths);
        let mut hmms = Vec::with_capacity(lengths);
        let mut state_spaces = Vec::with_capacity(lengths);
        for (index, beats) in beats_per_bar.iter().enumerate() {
            let min_interval = 60.0 * fps / max_bpm[index];
            let max_interval = 60.0 * fps / min_bpm[index];
            let state_space =
                BarStateSpace::new(*beats, min_interval, max_interval, Some(num_tempi[index]));
            let transition_model = BarTransitionModel::new(
                state_space.clone(),
                vec![transition_lambda[index]; *beats],
            );
            let observation_model =
                RNNDownBeatTrackingObservationModel::new(&state_space, observation_lambda);
            let hmm = HiddenMarkovModel::new(
                transition_model.into_transition_model(),
                observation_model,
                None,
            )
            .expect("the bar HMM starts from a uniform distribution");
            hmms.push(hmm);
            state_spaces.push(state_space);
        }
        Self {
            hmms,
            state_spaces,
            beats_per_bar: beats_per_bar.to_vec(),
            transition_lambda,
            observation_lambda,
            threshold,
            correct,
            fps,
        }
    }

    /// Build a processor with the reference's defaults.
    pub fn with_defaults(beats_per_bar: &[usize], fps: f64) -> Self {
        Self::new(
            beats_per_bar,
            &[MIN_BPM],
            &[MAX_BPM],
            &[NUM_TEMPI],
            &[Some(TRANSITION_LAMBDA)],
            OBSERVATION_LAMBDA,
            THRESHOLD,
            CORRECT,
            fps,
        )
    }

    /// The lambda of every bar length (the reference's `transition_lambda`).
    pub fn transition_lambda(&self) -> &[Option<f64>] {
        &self.transition_lambda
    }

    /// The splitting of a beat period into beat and downbeat parts.
    pub fn observation_lambda(&self) -> f64 {
        self.observation_lambda
    }

    /// The modelled bar lengths.
    pub fn beats_per_bar(&self) -> &[usize] {
        &self.beats_per_bar
    }

    /// The HMM of every bar length.
    pub fn hmms(&self) -> &[HiddenMarkovModel<RNNDownBeatTrackingObservationModel>] {
        &self.hmms
    }

    /// The state space of every bar length.
    pub fn state_spaces(&self) -> &[BarStateSpace] {
        &self.state_spaces
    }

    /// The frame rate.
    pub fn fps(&self) -> f64 {
        self.fps
    }

    /// Decode every HMM and return the results in bar-length order.
    pub fn viterbi_all(&self, activations: ArrayView2<'_, f64>) -> Vec<ViterbiResult> {
        self.hmms
            .iter()
            .map(|hmm| hmm.viterbi(activations))
            .collect()
    }

    /// Detect (down-)beats from two-column activations (beat, downbeat).
    pub fn process(&self, activations: ArrayView2<'_, f64>) -> Vec<DownBeat> {
        let mut first = 0usize;
        let activations = if self.threshold != 0.0 {
            let (segment, offset) = threshold_activations_2d(activations, self.threshold);
            first += offset;
            segment
        } else {
            activations.to_owned()
        };
        if activations.iter().all(|value| *value == 0.0) || activations.shape()[0] == 0 {
            return Vec::new();
        }
        let results = self.viterbi_all(activations.view());
        // the bar length with the highest path log probability wins (first one on ties)
        let mut best = 0usize;
        let mut best_probability = f64::NEG_INFINITY;
        for (index, result) in results.iter().enumerate() {
            if result.log_probability > best_probability {
                best_probability = result.log_probability;
                best = index;
            }
        }
        let path = &results[best].path;
        if path.is_empty() {
            return Vec::new();
        }
        let state_space = &self.state_spaces[best];
        let observation_model = self.hmms[best].observation_model();
        let positions = state_space.state_positions();
        let beat_numbers: Vec<i64> = path
            .iter()
            .map(|state| positions[*state as usize].trunc() as i64 + 1)
            .collect();
        let mut beats: Vec<usize> = Vec::new();
        if self.correct {
            let beat_range: Vec<bool> = path
                .iter()
                .map(|state| observation_model.pointers()[*state as usize] >= 1)
                .collect();
            if !beat_range.iter().any(|in_beat| *in_beat) {
                return Vec::new();
            }
            for (from, to) in regions(&beat_range) {
                // `np.argmax` over the *flattened* two-column window, divided by two:
                // the scan is element-wise in row-major order, so a later row wins on its
                // second column even when its first column is smaller. Comparing rows
                // lexicographically instead picks the wrong frame (0.08 instead of 0.09
                // on the reference sample).
                let window = activations.slice(ndarray::s![from..to, ..]);
                let mut best_flat = 0usize;
                let mut best_value = f64::NEG_INFINITY;
                for (flat, value) in window.iter().enumerate() {
                    if *value > best_value {
                        best_value = *value;
                        best_flat = flat;
                    }
                }
                beats.push(from + best_flat / activations.shape()[1]);
            }
        } else {
            // beats where the beat number changes
            for frame in 1..beat_numbers.len() {
                if beat_numbers[frame] != beat_numbers[frame - 1] {
                    beats.push(frame);
                }
            }
        }
        beats
            .into_iter()
            .map(|beat| DownBeat {
                time: (beat + first) as f64 / self.fps,
                beat_number: beat_numbers[beat],
            })
            .collect()
    }
}

/// Repeat a slice until it has `length` entries (the reference's `np.repeat`).
fn repeat_to<T: Clone>(values: &[T], length: usize) -> Vec<T> {
    assert!(!values.is_empty(), "at least one value is required");
    if values.len() == length {
        return values.to_vec();
    }
    let mut out = Vec::with_capacity(length);
    for index in 0..length {
        out.push(values[index % values.len()].clone());
    }
    out
}

/// Keep the rows in which any column reaches `threshold`, plus the first row index.
pub fn threshold_activations_2d(
    activations: ArrayView2<'_, f64>,
    threshold: f64,
) -> (Array2<f64>, usize) {
    let rows = activations.shape()[0];
    let first = (0..rows).find(|row| {
        activations
            .row(*row)
            .iter()
            .any(|value| *value >= threshold)
    });
    let last = (0..rows).rev().find(|row| {
        activations
            .row(*row)
            .iter()
            .any(|value| *value >= threshold)
    });
    match (first, last) {
        (Some(first), Some(last)) => (
            activations.slice(ndarray::s![first..=last, ..]).to_owned(),
            first,
        ),
        _ => (Array2::<f64>::zeros((0, activations.shape()[1])), 0),
    }
}

/// Contiguous `true` regions of a boolean mask as `(from, to)` half-open ranges.
fn regions(mask: &[bool]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (index, value) in mask.iter().enumerate() {
        match (value, start) {
            (true, None) => start = Some(index),
            (false, Some(from)) => {
                out.push((from, index));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(from) = start {
        out.push((from, mask.len()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_dimensional_thresholding_keeps_rows_with_any_column() {
        let activations = ndarray::array![
            [0.0, 0.0],
            [0.01, 0.06],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.07, 0.01],
            [0.0, 0.0],
        ];
        let (segment, first) = threshold_activations_2d(activations.view(), 0.05);
        assert_eq!(first, 1);
        // rows 1..=4 are kept, as in the reference (first to last row above threshold)
        assert_eq!(segment.shape(), &[4, 2]);
    }

    #[test]
    fn a_steady_four_four_pulse_gives_a_bar_of_beats() {
        // Exactly the input the reference was asked for while writing this test: 600
        // frames at 100 fps with a floor of 0.3 in both columns and a 0.9 peak every
        // 50 frames (120 bpm, inside the modelled intervals), the downbeat column
        // peaking on every fourth beat. madmom returns 13 beats with beat numbers
        // cycling 1, 2, 3, 4.
        let processor = DBNDownBeatTrackingProcessor::with_defaults(&[4], 100.0);
        let frames = 600;
        let mut activations = Array2::<f64>::from_elem((frames, 2), 0.3);
        for frame in (0..frames).step_by(50) {
            if frame % 200 == 0 {
                activations[[frame, 1]] = 0.9;
            } else {
                activations[[frame, 0]] = 0.9;
            }
        }
        let beats = processor.process(activations.view());
        assert_eq!(beats.len(), 13, "reference beats: {:?}", beats);
        assert_eq!(beats[0].beat_number, 1);
        assert!((beats[0].time - 0.0).abs() < 1e-9);
        assert!((beats[1].time - 0.5).abs() < 1e-9);
        let numbers: Vec<i64> = beats.iter().map(|beat| beat.beat_number).collect();
        assert_eq!(&numbers[..6], &[1, 2, 3, 4, 1, 2]);
        for number in &numbers {
            assert!((1..=4).contains(number), "beat number {number}");
        }
    }
}
