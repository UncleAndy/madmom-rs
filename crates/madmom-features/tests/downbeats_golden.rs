//! Joint beat/downbeat DBN against the reference dumps.
//!
//! Input is the reference's own downbeat activations (`sample.downbeats_blstm.npz`), so
//! the network is replaced by its output and what is verified is the DBN: the Viterbi
//! path of every modelled bar length (exact integer states), their log probabilities,
//! and the resulting (down-)beats with their position inside the bar.

use std::path::{Path, PathBuf};

use madmom_features::downbeats::DBNDownBeatTrackingProcessor;
use madmom_testkit::load_activations;

/// The bar lengths the dump used.
const BEATS_PER_BAR: [usize; 2] = [3, 4];

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

fn load(golden: &str) -> Option<(Vec<f32>, Vec<usize>)> {
    let (values, shape, _) = load_any(golden)?;
    Some((values, shape))
}

/// Like [`load`] but also returns the double precision values of a dump.
fn load_f64(golden: &str) -> Option<Vec<f64>> {
    let path = golden_dir().join("downbeats").join(golden);
    if !path.is_file() {
        assert!(
            std::env::var("MADMOM_REQUIRE_GOLDEN").is_err(),
            "{} is missing: run tools/dump_downbeats.py first",
            path.display()
        );
        return None;
    }
    let array = madmom_io::npy::read_npy(&path).expect("dump must be readable");
    Some(
        array
            .as_f64()
            .expect("log probabilities are stored as doubles"),
    )
}

fn load_any(golden: &str) -> Option<(Vec<f32>, Vec<usize>, ())> {
    let path = golden_dir().join("downbeats").join(golden);
    if !path.is_file() {
        assert!(
            std::env::var("MADMOM_REQUIRE_GOLDEN").is_err(),
            "{} is missing: run tools/dump_downbeats.py first (see tests/golden/README.md)",
            path.display()
        );
        return None;
    }
    let array = madmom_io::npy::read_npy(&path).expect("dump must be readable");
    let shape = array.shape().to_vec();
    Some((
        array.as_f32().expect("dumps hold numeric values"),
        shape,
        (),
    ))
}

#[test]
fn downbeats_match_the_reference_from_the_reference_activations() {
    let (activations, fps) = load_activations("activations/sample.downbeats_blstm.npz")
        .expect("the downbeat activations fixture must be readable");
    assert_eq!(activations.shape(), &[281, 2]);
    let fps = fps.expect("the fixture carries its frame rate");
    assert_eq!(fps, 100.0);

    let processor = DBNDownBeatTrackingProcessor::with_defaults(&BEATS_PER_BAR, fps);
    // the state spaces of the two bar lengths, as the reference builds them
    assert_eq!(processor.state_spaces()[0].num_states(), 11157);
    assert_eq!(processor.state_spaces()[1].num_states(), 14876);

    // every bar length decodes to the same path as the reference
    let matrix = activations
        .into_dimensionality::<ndarray::Ix2>()
        .expect("activations are (frames, 2)")
        .mapv(|value| value as f64);
    let results = processor.viterbi_all(matrix.view());
    assert_eq!(results.len(), BEATS_PER_BAR.len());
    for (index, beats_per_bar) in BEATS_PER_BAR.iter().enumerate() {
        let Some((expected_path, _)) = load(&format!("path_{beats_per_bar}.npy")) else {
            return;
        };
        let Some(expected_logprob) = load_f64(&format!("logprob_{beats_per_bar}.npy")) else {
            return;
        };
        assert_eq!(
            results[index].path.len(),
            expected_path.len(),
            "bar {beats_per_bar}: path length"
        );
        for (frame, (got, want)) in results[index]
            .path
            .iter()
            .zip(expected_path.iter())
            .enumerate()
        {
            assert_eq!(
                *got as f32, *want,
                "bar {beats_per_bar}: state path differs at frame {frame}"
            );
        }
        assert!(
            (results[index].log_probability - expected_logprob[0]).abs() < 1e-9,
            "bar {beats_per_bar}: log probability {} against {}",
            results[index].log_probability,
            expected_logprob[0]
        );
        eprintln!(
            "bar {beats_per_bar}: {} frames identical, log probability {:.6}",
            results[index].path.len(),
            results[index].log_probability
        );
    }

    // the thresholded stage the processor decodes: the trimmed activations and the
    // paths decoded from them (a separate check, because the thresholding shifts both
    // the input and the frame indices)
    let (trimmed_values, trimmed_shape) = load("trimmed.npy").expect("trimmed dump");
    let trimmed_reference = ndarray::Array2::from_shape_vec(
        (trimmed_shape[0], trimmed_shape[1]),
        trimmed_values.iter().map(|value| *value as f64).collect(),
    )
    .expect("trimmed dump shape");
    let (trimmed, first) =
        madmom_features::downbeats::threshold_activations_2d(matrix.view(), 0.05);
    assert_eq!(trimmed.shape(), trimmed_reference.shape(), "trimmed shape");
    assert_eq!(first, 7, "the reference drops the first seven rows");
    for (got, want) in trimmed.iter().zip(trimmed_reference.iter()) {
        assert_eq!(got, want, "trimmed activations differ");
    }
    let trimmed_results = processor.viterbi_all(trimmed.view());
    for (index, beats_per_bar) in BEATS_PER_BAR.iter().enumerate() {
        let Some((expected_path, _)) = load(&format!("path_trimmed_{beats_per_bar}.npy")) else {
            return;
        };
        for (frame, (got, want)) in trimmed_results[index]
            .path
            .iter()
            .zip(expected_path.iter())
            .enumerate()
        {
            assert_eq!(
                *got as f32, *want,
                "bar {beats_per_bar}: trimmed path differs at frame {frame}"
            );
        }
        eprintln!(
            "bar {beats_per_bar}: trimmed path {} frames identical",
            trimmed_results[index].path.len()
        );
    }

    // the (down-)beats themselves
    let Some((expected, shape)) = load("result.npy") else {
        return;
    };
    assert_eq!(shape, vec![8, 2]);
    let beats = processor.process(matrix.view());
    assert_eq!(
        beats.len(),
        shape[0],
        "number of (down-)beats differs: got {:?}, reference {:?}",
        beats,
        expected
    );
    for (index, beat) in beats.iter().enumerate() {
        let expected_time = expected[index * 2] as f64;
        let expected_number = expected[index * 2 + 1];
        assert!(
            (beat.time - expected_time).abs() <= 1e-6,
            "(down-)beat {index}: time {} against {expected_time}",
            beat.time
        );
        assert_eq!(
            beat.beat_number, expected_number as i64,
            "(down-)beat {index}: number"
        );
    }
    eprintln!(
        "(down-)beats: {:?}",
        beats
            .iter()
            .map(|beat| (beat.time, beat.beat_number))
            .collect::<Vec<_>>()
    );
}
