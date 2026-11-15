//! Beat tracking against the reference dumps.
//!
//! The activations come from madmom's own `BEATS_BLSTM` ensemble (`sample.activations.npy`,
//! dumped by `tools/dump_reference.py`), so the network itself is replaced by the
//! reference numbers here — "golden mode" from the plan. What is compared is everything
//! the port has to reproduce: the state path of the Viterbi decoding (exact integer
//! states, no tolerance) and the beat positions in seconds.

use std::path::{Path, PathBuf};

use madmom_features::beats::DBNBeatTrackingProcessor;

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

fn load(golden: &str) -> Option<Vec<f32>> {
    let path = golden_dir().join(golden);
    if !path.is_file() {
        assert!(
            std::env::var("MADMOM_REQUIRE_GOLDEN").is_err(),
            "{} is missing: generate the dumps first (see tests/golden/README.md)",
            path.display()
        );
        return None;
    }
    let array = madmom_io::npy::read_npy(&path).expect("dump must be readable");
    Some(array.as_f32().expect("dumps hold numeric values"))
}

#[test]
fn beats_match_the_reference_from_the_reference_activations() {
    let Some(activations) = load("sample.activations.npy") else {
        return;
    };
    let Some(expected_path) = load("sample.path.npy") else {
        return;
    };
    let Some(expected_beats) = load("sample.beats.npy") else {
        return;
    };

    let activations: Vec<f64> = activations.iter().map(|value| *value as f64).collect();
    // the dump used DBNBeatTrackingProcessor(fps=100, min_bpm=55, max_bpm=215)
    let processor = DBNBeatTrackingProcessor::with_defaults(100.0);
    assert_eq!(processor.state_space().num_states(), (28 + 109) * 82 / 2);

    let result = processor.viterbi_path(&activations);
    assert_eq!(
        result.path.len(),
        expected_path.len(),
        "the path must cover every frame"
    );
    for (frame, (got, want)) in result.path.iter().zip(expected_path.iter()).enumerate() {
        assert_eq!(
            *got as f32, *want,
            "state path differs at frame {frame}: got {got}, reference {want}"
        );
    }
    eprintln!(
        "path: {} frames identical, log probability {:.6}",
        result.path.len(),
        result.log_probability
    );

    let beats = processor
        .process_offline(&activations)
        .expect("offline beat tracking must run");
    assert_eq!(
        beats.len(),
        expected_beats.len(),
        "number of beats differs: got {:?}, reference {:?}",
        beats,
        expected_beats
    );
    for (index, (got, want)) in beats.iter().zip(expected_beats.iter()).enumerate() {
        assert!(
            (got - *want as f64).abs() <= 1e-6,
            "beat {index} differs: got {got}, reference {want}"
        );
    }
    eprintln!("beats: {:?} (all within 1e-6 of the reference)", beats);
}
