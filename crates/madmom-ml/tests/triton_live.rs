#![cfg(feature = "triton-live")]
//! End-to-end check of the in-process engine: the server starts, the ONNX Runtime backend
//! loads a model from a repository, a request goes through and the activations come back.
//!
//! The model is the small 5x5 `add_with_initializer` shipped with the ONNX Runtime backend
//! sources (a submodule), so nothing has to be generated first. Values are deliberately not
//! asserted: the point is that the engine works, which is what the parity tests in
//! madmom-features then rely on.
//!
//! Building and running this needs the runtime from `tools/build_triton_core.sh` and
//! `tools/build_onnx_backend.sh`:
//!
//! ```text
//! RUSTFLAGS="-L $PWD/third_party/triton-runtime/lib64" \
//! LD_LIBRARY_PATH="$PWD/third_party/triton-runtime/lib64" \
//! cargo test -p madmom-ml --features triton-live
//! ```

use std::path::PathBuf;

use madmom_ml::inference::{ActivationModel, InputShape};
use madmom_ml::triton::{TritonConfig, TritonModel};
use ndarray::{ArrayD, IxDyn};

/// Repository holding the example model, and the backends we built.
fn paths() -> (PathBuf, PathBuf) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    (
        root.join("third_party/onnxruntime_backend/test/initializer_as_input/models"),
        root.join("third_party/triton-runtime/backends"),
    )
}

#[test]
fn serves_an_onnx_model_end_to_end() {
    let (repository, backends) = paths();
    let config = TritonConfig::new(&repository).with_backend_directory(&backends);
    let mut model = TritonModel::start(
        &config,
        "add_with_initializer",
        "INPUT",
        "OUTPUT",
        InputShape::Frames(5),
    )
    .expect("the server starts and loads the model");

    assert_eq!(model.input_shape(), InputShape::Frames(5));
    let values: Vec<f32> = (0..25).map(|i| i as f32).collect();
    let features = ArrayD::from_shape_vec(IxDyn(&[5, 5]), values).expect("5x5 fits 25 values");

    let activations = model.predict(&features).expect("one inference succeeds");
    assert_eq!(activations.shape(), &[5, 5]);
    assert!(
        activations.iter().all(|v| v.is_finite()),
        "activations are finite: {activations:?}"
    );
}
