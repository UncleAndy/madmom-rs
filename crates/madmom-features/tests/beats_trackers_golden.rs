//! The beat trackers against the reference dumps.
//!
//! `beats_golden.rs` pins the DBN's default offline path; this file covers the rest of
//! `madmom/features/beats.py`: the peak correction switched off, the thresholds, the empty
//! path, the forward (online) mode — all-at-once and frame by frame, with and without a
//! reset — the local- and global-tempo trackers, and the CRF.
//!
//! Everything runs on madmom's own activation fixtures, and the DBN's output is
//! additionally required to equal the `detections/sample.dbn_beat_tracker.txt` the reference
//! command line program wrote — that is what makes the fixture activations the ones that
//! file came from.

use std::path::{Path, PathBuf};

use madmom_features::beats::{
    BeatDetectionProcessor, BeatTrackingProcessor, DBNBeatTrackingProcessor,
};
use madmom_features::beats_crf::{CRFBeatDetectionProcessor, INTERVAL_SIGMA};
use madmom_testkit::load_activations;

const DETECTIONS_ATOL: f64 = 1e-6;

fn golden_dir() -> PathBuf {
    std::env::var("MADMOM_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace_root().join("tests").join("golden"))
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate lives in <root>/crates/madmom-features")
        .to_path_buf()
}

fn load(tag: &str) -> Option<Vec<f64>> {
    let path = golden_dir().join("beats").join(format!("{tag}.npy"));
    if !path.is_file() {
        assert!(
            std::env::var("MADMOM_REQUIRE_GOLDEN").is_err(),
            "{} is missing: run tools/dump_beats.py first (see tests/golden/README.md)",
            path.display()
        );
        return None;
    }
    let array = madmom_io::npy::read_npy(&path).expect("dump must be readable");
    Some(array.as_f64().expect("beats dumps hold doubles"))
}

fn assert_beats(got: &[f64], tag: &str) -> bool {
    let Some(expected) = load(tag) else {
        return false;
    };
    assert_eq!(
        got.len(),
        expected.len(),
        "{tag}: {} beats against {}: {got:?} against {expected:?}",
        got.len(),
        expected.len()
    );
    for (index, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - expected).abs() <= DETECTIONS_ATOL,
            "{tag}: beat {index} is {got} against {expected}"
        );
    }
    eprintln!("{tag}: {} beats identical", got.len());
    true
}

/// The beat activation fixture (100 fps) the offline trackers use.
fn blstm() -> Vec<f64> {
    let (activations, fps) = load_activations("activations/sample.beats_blstm.npz")
        .expect("the beat activations fixture must be readable");
    assert_eq!(fps, Some(100.0));
    activations.iter().map(|value| *value as f64).collect()
}

/// The online (uni-directional LSTM) fixture the forward mode uses.
fn lstm() -> Vec<f64> {
    let (activations, fps) = load_activations("activations/sample.beats_lstm.npz")
        .expect("the online beat activations fixture must be readable");
    assert_eq!(fps, Some(100.0));
    activations.iter().map(|value| *value as f64).collect()
}

#[test]
fn dbn_tracker_variants_match_the_reference() {
    let activations = blstm();
    if load("dbn_default").is_none() {
        return;
    }

    let processor = DBNBeatTrackingProcessor::with_defaults(100.0);
    let beats = processor.process_offline(&activations).expect("beats");
    assert_beats(&beats, "dbn_default");
    // the command line program's own file, which is the same activations
    assert_beats(&beats, "det_dbn");

    // beats on the first and last frame of the signal
    let mut edge = vec![1e-4; 200];
    for frame in [0, 49, 99, 149, 199] {
        edge[frame] = 1.0;
    }
    let beats = processor.process_offline(&edge).expect("edge beats");
    assert_beats(&beats, "dbn_edge_frames");

    // without the peak correction: the local minima of the state positions
    let build = |threshold: f64, correct: bool| {
        DBNBeatTrackingProcessor::new(
            madmom_features::beats::MIN_BPM,
            madmom_features::beats::MAX_BPM,
            None,
            madmom_features::beats::TRANSITION_LAMBDA,
            madmom_features::beats::OBSERVATION_LAMBDA,
            threshold,
            correct,
            100.0,
        )
    };
    let beats = build(0.0, false)
        .process_offline(&activations)
        .expect("beats without correction");
    assert_beats(&beats, "dbn_no_correction");

    // the threshold cuts the activations before the decoding
    let beats = build(0.25, true)
        .process_offline(&activations)
        .expect("beats with a threshold");
    assert_beats(&beats, "dbn_threshold_025");
    let beats = build(1.0, true)
        .process_offline(&activations)
        .expect("no beats above the threshold");
    assert!(beats.is_empty(), "threshold 1 leaves nothing: {beats:?}");
    assert_beats(&beats, "dbn_threshold_1");

    // an activation that leads to an empty path
    let alternating = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let beats = processor
        .process_offline(&alternating)
        .expect("the empty path case");
    assert_beats(&beats, "dbn_empty_path");
    assert!(beats.is_empty(), "the reference returns no beats here");
}

#[test]
fn dbn_forward_mode_matches_the_reference() {
    let activations = lstm();
    if load("dbn_forward_all").is_none() {
        return;
    }
    let mut processor = DBNBeatTrackingProcessor::with_defaults(100.0);
    let beats = processor
        .process_forward(&activations, true)
        .expect("forward beats");
    assert_beats(&beats, "dbn_forward_all");

    // frame by frame, starting from a reset state
    processor.reset();
    let mut frames: Vec<f64> = Vec::new();
    for (frame, activation) in activations.iter().enumerate() {
        let beats = processor
            .process_forward(&[*activation], false)
            .expect("forward, frame by frame");
        if !beats.is_empty() {
            frames.push(frame as f64);
        }
    }
    assert_beats(&frames, "dbn_forward_frames");

    // and without the reset the recursion continues, which reports different frames
    let mut frames: Vec<f64> = Vec::new();
    for (frame, activation) in activations.iter().enumerate() {
        let beats = processor
            .process_forward(&[*activation], false)
            .expect("forward, frame by frame");
        if !beats.is_empty() {
            frames.push(frame as f64);
        }
    }
    assert_beats(&frames, "dbn_forward_frames_noreset");
}

#[test]
fn local_and_global_tempo_trackers_match_the_reference() {
    let activations = blstm();
    if load("tracker").is_none() {
        return;
    }
    let tracker =
        BeatTrackingProcessor::with_defaults(Some(madmom_features::beats::LOOK_AHEAD), 100.0);
    let beats = tracker.process(&activations).expect("tracked beats");
    assert_beats(&beats, "tracker");

    let detector = BeatDetectionProcessor::with_defaults(100.0);
    let beats = detector.process(&activations).expect("detected beats");
    assert_beats(&beats, "detector");
}

#[test]
fn crf_beat_detection_matches_the_reference() {
    let activations = blstm();
    if load("crf").is_none() {
        return;
    }
    let processor = CRFBeatDetectionProcessor::with_defaults(100.0);
    assert_eq!(processor.interval_sigma(), INTERVAL_SIGMA);
    let intervals: Vec<f64> = processor
        .possible_intervals(&activations)
        .expect("candidate intervals")
        .iter()
        .map(|interval| *interval as f64)
        .collect();
    let expected_intervals = load("crf_intervals").expect("interval dump");
    assert_eq!(intervals, expected_intervals, "candidate intervals");
    let beats = processor.process(&activations).expect("crf beats");
    assert_beats(&beats, "crf");
}
