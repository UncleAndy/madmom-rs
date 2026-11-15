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

use madmom_audio::signal::{signal_frame_f64, smooth_f64};
use madmom_core::scipy::{argrelmin, Mode};
use madmom_ml::hmm::{HiddenMarkovModel, ObservationModel, ViterbiResult};

use crate::tempo::{
    dominant_interval, CombFilterTempoHistogramProcessor, TempoError, TempoHistogramSource,
    ACT_SMOOTH, HIST_SMOOTH,
};
use ndarray::{Array2, ArrayView1};

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
/// Fraction of a beat interval searched to either side of an assumed beat position.
pub const LOOK_ASIDE: f64 = 0.2;
/// Seconds looked at in both directions to determine the local tempo.
pub const LOOK_AHEAD: f64 = 10.0;

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

/// Detect beats by looking for the nearest activation peak around an assumed position.
///
/// Port of `detect_beats` (the 2013 tracking approach): starting from every position inside
/// the first beat interval, the beat positions are followed recursively — at each step a
/// Hamming-weighted window around the assumed position is searched for its maximum — and the
/// start whose beats carry the most activation wins.
///
/// Two details matter: the window is centred on the assumed position (`signal_frame` with a
/// hop of 1), so a peak can pull the beat to either side, and the positions are **signed**:
/// near the start of the signal a beat can end up before frame 0, where the reference's
/// `activations[positions]` then wraps around to the end of the array (Python's negative
/// indexing). Both are reproduced here.
pub fn detect_beats(activations: &[f64], interval: usize, look_aside: f64) -> Vec<i64> {
    let frames_look_aside = std::cmp::max(1, (interval as f64 * look_aside) as usize);
    let window = madmom_audio::signal::hamming_f64(2 * frames_look_aside);
    let length = activations.len() as i64;

    let mut positions: Vec<i64> = Vec::new();
    // the reference recurses; an explicit loop produces the same sequence without the
    // recursion limit it has to raise for long signals
    let detect = |start: i64, positions: &mut Vec<i64>| {
        let mut position = start;
        loop {
            let frame = signal_frame_f64(
                ArrayView1::from(activations),
                position,
                frames_look_aside * 2,
                1.0,
                0,
                0.0,
            );
            let weighted = frame * &window;
            // np.argmax: the first maximum wins, and only a maximum beyond the first
            // sample moves the beat
            let mut best = 0usize;
            for (index, value) in weighted.iter().enumerate() {
                if *value > weighted[best] {
                    best = index;
                }
            }
            if best > 0 {
                position = best as i64 + position - frames_look_aside as i64;
            }
            positions.push(position);
            if position + (interval as i64) < length {
                position += interval as i64;
            } else {
                return;
            }
        }
    };

    // try every start position inside the first interval and keep the one whose beats sum
    // up to the most activation
    let mut sums = vec![0.0f64; interval.max(1)];
    for (start, sum) in sums.iter_mut().enumerate() {
        positions.clear();
        detect(start as i64, &mut positions);
        *sum = positions
            .iter()
            .map(|position| activations[wrap_index(*position, length)])
            .sum();
    }
    let mut start_position = 0usize;
    for (index, value) in sums.iter().enumerate() {
        if *value > sums[start_position] {
            start_position = index;
        }
    }
    positions.clear();
    detect(start_position as i64, &mut positions);
    positions
}

/// Python's negative indexing: `-1` is the last element.
fn wrap_index(index: i64, length: i64) -> usize {
    index.rem_euclid(length) as usize
}

/// The reference's `BeatTrackingProcessor`: beat tracking with a local tempo estimate.
///
/// With `look_ahead` set (the default) the tempo is re-estimated every `look_ahead` seconds
/// from a window around the current position, so the beats follow a changing tempo; with
/// `look_ahead = None` (the [`BeatDetectionProcessor`]) a single global interval is used.
#[derive(Debug, Clone)]
pub struct BeatTrackingProcessor {
    look_aside: f64,
    look_ahead: Option<f64>,
    fps: f64,
    histogram: CombFilterTempoHistogramProcessor,
    hist_smooth: usize,
    act_smooth: f64,
}

impl BeatTrackingProcessor {
    /// Build a processor.
    pub fn new(
        look_aside: f64,
        look_ahead: Option<f64>,
        fps: f64,
        histogram: CombFilterTempoHistogramProcessor,
        hist_smooth: usize,
        act_smooth: f64,
    ) -> Self {
        Self {
            look_aside,
            look_ahead,
            fps,
            histogram,
            hist_smooth,
            act_smooth,
        }
    }

    /// Build a processor with the reference's defaults.
    pub fn with_defaults(look_ahead: Option<f64>, fps: f64) -> Self {
        Self::new(
            LOOK_ASIDE,
            look_ahead,
            fps,
            CombFilterTempoHistogramProcessor::with_defaults(fps),
            HIST_SMOOTH,
            ACT_SMOOTH,
        )
    }

    /// Detect beat positions [seconds].
    pub fn process(&self, activations: &[f64]) -> Result<Vec<f64>, TempoError> {
        // the reference smooths with int(fps * act_smooth) frames, i.e. 14 at 100 fps
        let act_smooth = (self.fps * self.act_smooth) as usize;
        let activations = smooth_f64(ArrayView1::from(activations), act_smooth)?;
        let mut detections: Vec<i64> = Vec::new();
        match self.look_ahead {
            // a single global tempo for the whole signal
            None => {
                let histogram = self.histogram.histogram(activations.view())?;
                let interval = dominant_interval(&histogram, Some(self.hist_smooth))? as usize;
                detections = detect_beats(
                    activations
                        .as_slice()
                        .expect("the activations are contiguous"),
                    interval,
                    self.look_aside,
                );
            }
            Some(look_ahead) => {
                let look_ahead_frames = (look_ahead * self.fps) as i64;
                let mut pos: i64 = 0;
                while (pos as usize) < activations.len() {
                    // the local tempo comes from a window around the current position
                    let window = signal_frame_f64(
                        activations.view(),
                        pos,
                        (look_ahead_frames * 2) as usize,
                        1.0,
                        0,
                        0.0,
                    );
                    let histogram = self.histogram.histogram(window.view())?;
                    let interval = dominant_interval(&histogram, Some(self.hist_smooth))? as usize;
                    let mut positions = detect_beats(
                        window.as_slice().expect("the window is contiguous"),
                        interval,
                        self.look_aside,
                    );
                    for position in positions.iter_mut() {
                        *position += pos - look_ahead_frames;
                    }
                    let next_pos = match detections.last() {
                        Some(last) => last + self.histogram.min_interval() as i64,
                        None => 0,
                    };
                    positions.retain(|position| *position >= next_pos);
                    if positions.is_empty() {
                        return Err(TempoError::NoBeats);
                    }
                    // take the detected beat closest to the assumed position
                    let mut closest = positions[0];
                    for position in positions.iter() {
                        if (*position - pos).abs() < (closest - pos).abs() {
                            closest = *position;
                        }
                    }
                    pos = closest;
                    detections.push(pos);
                    pos += interval as i64;
                }
            }
        }
        // convert to seconds and drop the beats before the signal
        Ok(detections
            .into_iter()
            .map(|frame| frame as f64 / self.fps)
            .filter(|seconds| *seconds >= 0.0)
            .collect())
    }
}

/// The reference's `BeatDetectionProcessor`: the same tracker with a global tempo.
#[derive(Debug, Clone)]
pub struct BeatDetectionProcessor {
    tracker: BeatTrackingProcessor,
}

impl BeatDetectionProcessor {
    /// Build a processor.
    pub fn new(look_aside: f64, fps: f64, histogram: CombFilterTempoHistogramProcessor) -> Self {
        Self {
            tracker: BeatTrackingProcessor::new(
                look_aside,
                None,
                fps,
                histogram,
                HIST_SMOOTH,
                ACT_SMOOTH,
            ),
        }
    }

    /// Build a processor with the reference's defaults.
    pub fn with_defaults(fps: f64) -> Self {
        Self::new(
            LOOK_ASIDE,
            fps,
            CombFilterTempoHistogramProcessor::with_defaults(fps),
        )
    }

    /// Detect beat positions [seconds].
    pub fn process(&self, activations: &[f64]) -> Result<Vec<f64>, TempoError> {
        self.tracker.process(activations)
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
    /// Online state (the reference keeps the same fields when `online=True`).
    counter: i64,
    last_beat: f64,
    tempo: f64,
    strength: f64,
    beat_counter: i32,
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
            counter: 0,
            last_beat: 0.0,
            tempo: 0.0,
            strength: 0.0,
            beat_counter: 0,
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
            // no peak correction: take the frames with the smallest beat state value, i.e.
            // the local minima of the state positions along the path, and keep only those
            // the observation model calls a beat
            let positions = self.state_space.state_positions();
            let path_positions: Vec<f64> = result
                .path
                .iter()
                .map(|state| positions[*state as usize])
                .collect();
            let minima = argrelmin(&path_positions, 1, Mode::Wrap)
                .map_err(|_| "argrelmin failed on the state positions")?;
            let pointers = self.hmm.observation_model().pointers();
            beats = minima
                .into_iter()
                .filter(|frame| pointers[result.path[*frame] as usize] == 1)
                .collect();
        }
        Ok(beats
            .into_iter()
            .map(|beat| (beat + first) as f64 / self.fps)
            .collect())
    }
}

impl DBNBeatTrackingProcessor {
    /// Reset the online state (the HMM's forward variables included).
    pub fn reset(&mut self) {
        self.hmm.reset();
        self.counter = 0;
        self.last_beat = 0.0;
        self.tempo = 0.0;
        self.strength = 0.0;
        self.beat_counter = 0;
    }

    /// Track beats causally with the HMM's forward variables (madmom's `process_online`,
    /// which is also exported as `process_forward`).
    ///
    /// Instead of decoding a path, every frame takes the state with the highest forward
    /// variable; frames in the beat part of the state space are candidates, and a candidate
    /// is only reported if it is at least `60 / max_bpm` seconds after the last reported
    /// beat. That rule is what keeps the forward path from reporting the clusters of
    /// neighbouring beats it otherwise produces.
    pub fn process_online(
        &mut self,
        activations: &[f64],
        reset: bool,
    ) -> Result<Vec<f64>, &'static str> {
        let matrix = Array2::from_shape_vec((activations.len(), 1), activations.to_vec())
            .expect("one column holds all activations");
        let forward = self.hmm.forward(matrix.view(), reset);
        let num_states = self.state_space.num_states();
        let pointers = self.hmm.observation_model().pointers();
        let mut beats: Vec<f64> = Vec::new();
        for frame in 0..activations.len() {
            // argmax over the states of this frame; the first maximum wins
            let mut best_state = 0usize;
            let mut best = f64::NEG_INFINITY;
            for state in 0..num_states {
                if forward[[frame, state]] > best {
                    best = forward[[frame, state]];
                    best_state = state;
                }
            }
            if pointers[best_state] != 1 {
                continue;
            }
            let current_beat = (frame as i64 + self.counter) as f64 / self.fps;
            let next_beat = self.last_beat + 60.0 / self.max_bpm;
            if current_beat >= next_beat {
                self.tempo = 60.0 / (current_beat - self.last_beat);
                self.last_beat = current_beat;
                beats.push(current_beat);
            }
        }
        self.counter += activations.len() as i64;
        Ok(beats)
    }

    /// The reference's alias for the online mode.
    pub fn process_forward(
        &mut self,
        activations: &[f64],
        reset: bool,
    ) -> Result<Vec<f64>, &'static str> {
        self.process_online(activations, reset)
    }

    /// The last tempo reported by the online mode [bpm].
    pub fn tempo(&self) -> f64 {
        self.tempo
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
