//! Dev-only helpers for parity tests.
//!
//! Everything here exists to make one sentence testable: *"this Rust stage
//! produces the same numbers as madmom"*. The tolerance constants mirror the
//! policy in the porting plan so that a test cannot quietly relax a threshold:
//!
//! | stage | tolerance |
//! |---|---|
//! | DSP in `f64` (windows, filterbanks) | [`DSP_ATOL`] |
//! | STFT / spectrograms in `f32` | [`SPEC_ATOL`] |
//! | neural network activations (float32) | [`NN_ATOL`] |
//! | detections / timestamps in seconds | [`DETECTION_ATOL`] |

#![deny(missing_docs)]

use std::path::PathBuf;

use madmom_io::npy::{self, NpyError};
use ndarray::{ArrayD, IxDyn};

/// Absolute tolerance for `f64` DSP stages (windows, filterbanks, differences).
pub const DSP_ATOL: f32 = 1e-6;
/// Relative tolerance for `f64` DSP stages.
pub const DSP_RTOL: f32 = 1e-9;
/// Absolute tolerance for `f32` spectral stages (STFT, spectrograms).
pub const SPEC_ATOL: f32 = 1e-5;
/// Absolute tolerance for neural network activations (float32).
pub const NN_ATOL: f32 = 1e-4;
/// Tolerance for timestamps in seconds (detections, annotations).
pub const DETECTION_ATOL: f32 = 1e-6;

/// Directory holding the madmom parity fixtures.
///
/// `$MADMOM_FIXTURES` wins, then `tests/fixtures` next to the crate, then the
/// submodule path `third_party/madmom/tests/data`.
pub fn fixtures_dir() -> PathBuf {
    npy::fixtures_dir()
}

/// Resolves a path inside the fixture directory.
pub fn fixture(relative: &str) -> PathBuf {
    fixtures_dir().join(relative)
}

/// Loads an `activations` member of a fixture `.npz` together with its `fps`.
///
/// A missing or object-typed `fps` member yields `None`, exactly like madmom's
/// `Activations.load` degrades when the frame rate cannot be read.
pub fn load_activations(relative: &str) -> Result<(ArrayD<f32>, Option<f64>), NpyError> {
    let path = fixture(relative);
    let activations = npy::read_npz_member(&path, "activations")?;
    let fps = match npy::read_npz_member(&path, "fps") {
        Ok(arr) => match arr.scalar_f64() {
            Ok(v) => Some(v),
            Err(NpyError::ObjectArray) => None,
            Err(e) => return Err(e),
        },
        Err(NpyError::MissingMember(_)) => None,
        Err(e) => return Err(e),
    };
    let shape: Vec<usize> = activations.header.shape.clone();
    let values = activations.as_f32()?;
    let array = ArrayD::from_shape_vec(IxDyn(&shape), values)
        .expect("fixture shape must match the number of values read");
    Ok((array, fps))
}

/// Largest absolute element-wise difference between two arrays.
///
/// Panics if the shapes differ: a shape mismatch is a bug in the test, not a
/// numerical difference.
pub fn max_abs_diff(actual: &ArrayD<f32>, expected: &ArrayD<f32>) -> f32 {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "shape mismatch: {:?} vs {:?}",
        actual.shape(),
        expected.shape()
    );
    actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Cosine similarity, a scale-free check for activations.
pub fn cosine_similarity(actual: &ArrayD<f32>, expected: &ArrayD<f32>) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (a, b) in actual.iter().zip(expected.iter()) {
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64) * (*a as f64);
        nb += (*b as f64) * (*b as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// Asserts that two arrays agree within `atol` (absolute) or `rtol`
/// (relative to the expected value), reporting the worst element on failure.
pub fn assert_array_close(actual: &ArrayD<f32>, expected: &ArrayD<f32>, atol: f32, rtol: f32) {
    assert_array_close_labeled("array", actual, expected, atol, rtol)
}

/// [`assert_array_close`] with a label naming the stage under test.
pub fn assert_array_close_labeled(
    label: &str,
    actual: &ArrayD<f32>,
    expected: &ArrayD<f32>,
    atol: f32,
    rtol: f32,
) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{}: shape mismatch {:?} vs {:?}",
        label,
        actual.shape(),
        expected.shape()
    );
    let mut worst = 0.0f32;
    let mut worst_index = 0usize;
    for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - b).abs();
        let allowed = atol + rtol * b.abs();
        if diff - allowed > worst {
            worst = diff - allowed;
            worst_index = i;
        }
    }
    if worst > 0.0 {
        let flat_actual = actual.iter().copied().collect::<Vec<_>>();
        let flat_expected = expected.iter().copied().collect::<Vec<_>>();
        panic!(
            "{}: {} differs from the reference beyond atol={} rtol={}:\n  \
             index {}: got {}, expected {} (|diff| = {})",
            label,
            actual
                .shape()
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x"),
            atol,
            rtol,
            worst_index,
            flat_actual[worst_index],
            flat_expected[worst_index],
            (flat_actual[worst_index] - flat_expected[worst_index]).abs(),
        );
    }
}

/// Asserts that two detection sequences (frame indices or seconds) match
/// element-wise within [`DETECTION_ATOL`].
pub fn assert_detections_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "detection count differs: {} vs {} (got first {}, expected first {})",
        actual.len(),
        expected.len(),
        actual.first().copied().unwrap_or(f64::NAN),
        expected.first().copied().unwrap_or(f64::NAN),
    );
    for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() <= DETECTION_ATOL as f64,
            "detection {} differs: {} vs {}",
            i,
            a,
            b
        );
    }
}
