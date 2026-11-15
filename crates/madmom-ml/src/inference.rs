//! Inference sources for the neural network stages.
//!
//! The port does not reimplement madmom's network layers. Activations come
//! from one of two places behind a single trait:
//!
//! - [`GoldenFiles`] — activations recorded from the reference implementation
//!   (the fixtures in `third_party/madmom/tests/data/activations`). This is
//!   what keeps beats/onsets/tempo development and tests independent of any
//!   model repository or Triton build.
//! - [`TritonLocal`] — the in-process Triton Inference Server via
//!   `tritonserver-rs`, running converted ONNX models. Enabled with the
//!   `triton-live` feature and implemented in Task 2.3 of the plan.
//!
//! Both only ever deal with the *interface* between local preprocessing and a
//! network: features in, activations out. Everything around it (multi
//! resolution stacking, ensemble averaging, peak picking, HMM decoding) stays
//! in Rust, which is why [`InputShape`] is part of the contract.

#![allow(clippy::result_large_err)]

use std::path::{Path, PathBuf};

use madmom_io::npy::{self, NpyError};
use ndarray::{ArrayD, IxDyn};
use thiserror::Error;

/// Errors of the inference layer.
#[derive(Debug, Error)]
pub enum InferenceError {
    /// A fixture could not be read.
    #[error(transparent)]
    Npy(#[from] NpyError),
    /// The feature matrix does not have the shape the model expects.
    #[error("model expects {} but got features {:?}", expected, got)]
    ShapeMismatch {
        /// Shape declared by the model.
        expected: InputShape,
        /// Shape of the supplied features.
        got: Vec<usize>,
    },
    /// The model repository or a specific model is missing.
    #[error("model {model} not found in repository {repository}")]
    MissingModel {
        /// Repository path.
        repository: PathBuf,
        /// Model name.
        model: String,
    },
    /// Asked for an inference path that is not implemented yet.
    #[error("not implemented yet: {0} (see the porting plan, phase 2)")]
    NotImplemented(&'static str),
    /// The in-process Triton bindings were not compiled in.
    #[error("the triton-live feature is required for in-process inference")]
    TritonFeatureDisabled,
    /// The in-process Triton server failed.
    #[error("triton inference for {model} failed: {cause}")]
    Triton {
        /// Model the request was made for.
        model: String,
        /// What the bindings reported.
        cause: String,
    },
}

impl InferenceError {
    /// Wraps a failure reported by the Triton bindings.
    #[cfg(feature = "triton-live")]
    pub(crate) fn triton<E: std::fmt::Display>(model: &str, cause: E) -> Self {
        Self::Triton {
            model: model.to_owned(),
            cause: cause.to_string(),
        }
    }
}

/// Shape of the feature matrix a model consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputShape {
    /// `(frames, features)` — feed-forward, recurrent and GRU models.
    Frames(usize),
    /// `(frames, bins, features)` for convolutions.
    FramesFreq(usize, usize),
    /// `(frames, bins, channels)` for stacked multi resolution CNNs.
    FramesFreqCh(usize, usize, usize),
}

impl std::fmt::Display for InputShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InputShape::Frames(n) => write!(f, "(frames, {})", n),
            InputShape::FramesFreq(b, n) => write!(f, "(frames, {}, {})", b, n),
            InputShape::FramesFreqCh(b, c, n) => write!(f, "(frames, {}, {}, {})", b, c, n),
        }
    }
}

/// A model that turns feature frames into activations.
pub trait ActivationModel {
    /// Shape of the input the model expects.
    fn input_shape(&self) -> InputShape;

    /// Runs inference on a feature matrix.
    fn predict(&mut self, features: &ArrayD<f32>) -> Result<ArrayD<f32>, InferenceError>;

    /// Frame rate of the activations, when the source knows it.
    fn fps(&self) -> Option<f64> {
        None
    }
}

/// Activations recorded from the reference implementation.
///
/// [`ActivationModel::predict`] validates the row count of the input but
/// returns the recorded activations: in golden mode the network is *replaced*,
/// not simulated. Tests that need the real network run with the `triton-live`
/// feature (plan Task 2.5).
#[derive(Debug, Clone)]
pub struct GoldenFiles {
    path: PathBuf,
    activations: ArrayD<f32>,
    fps: Option<f64>,
    expected_frames: Option<usize>,
}

impl GoldenFiles {
    /// Loads the `activations` (and optional `fps`) member of a fixture `.npz`.
    pub fn from_npz<P: AsRef<Path>>(path: P) -> Result<Self, InferenceError> {
        let path = path.as_ref().to_path_buf();
        let activations = npy::read_npz_member(&path, "activations")?;
        let shape = activations.header.shape.clone();
        let values = activations.as_f32()?;
        let array = ArrayD::from_shape_vec(IxDyn(&shape), values)
            .expect("activations member shape matches its data length");
        let fps = match npy::read_npz_member(&path, "fps") {
            Ok(arr) => arr.scalar_f64().ok(),
            Err(NpyError::MissingMember(_)) | Err(NpyError::ObjectArray) => None,
            Err(e) => return Err(e.into()),
        };
        let expected_frames = shape.first().copied();
        Ok(Self {
            path,
            activations: array,
            fps,
            expected_frames,
        })
    }

    /// Recorded activations.
    pub fn activations(&self) -> &ArrayD<f32> {
        &self.activations
    }

    /// Path the activations were read from.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl ActivationModel for GoldenFiles {
    fn input_shape(&self) -> InputShape {
        InputShape::Frames(0)
    }

    fn predict(&mut self, features: &ArrayD<f32>) -> Result<ArrayD<f32>, InferenceError> {
        if let Some(expected) = self.expected_frames {
            let got = features.shape().first().copied().unwrap_or(0);
            if got != 0 && got != expected {
                return Err(InferenceError::ShapeMismatch {
                    expected: InputShape::Frames(expected),
                    got: features.shape().to_vec(),
                });
            }
        }
        Ok(self.activations.clone())
    }

    fn fps(&self) -> Option<f64> {
        self.fps
    }
}

/// Where activations come from.
#[derive(Debug)]
pub enum ActivationSource {
    /// Recorded reference activations (development and tests).
    GoldenFiles(GoldenFiles),
    /// In-process Triton inference on ONNX models (requires `triton-live`).
    #[cfg(feature = "triton-live")]
    TritonLocal(TritonLocal),
}

impl ActivationModel for ActivationSource {
    fn input_shape(&self) -> InputShape {
        match self {
            ActivationSource::GoldenFiles(g) => g.input_shape(),
            #[cfg(feature = "triton-live")]
            ActivationSource::TritonLocal(t) => t.input_shape(),
        }
    }

    fn predict(&mut self, features: &ArrayD<f32>) -> Result<ArrayD<f32>, InferenceError> {
        match self {
            ActivationSource::GoldenFiles(g) => g.predict(features),
            #[cfg(feature = "triton-live")]
            ActivationSource::TritonLocal(t) => t.predict(features),
        }
    }

    fn fps(&self) -> Option<f64> {
        match self {
            ActivationSource::GoldenFiles(g) => g.fps(),
            #[cfg(feature = "triton-live")]
            ActivationSource::TritonLocal(t) => t.fps(),
        }
    }
}

/// In-process Triton model.
///
/// Placeholder: the fields (server handle, model name and version, tensor
/// names, tokio runtime) and the request plumbing land in Task 2.3 of the
/// porting plan. It returns [`InferenceError::NotImplemented`] instead of
/// panicking so that enabling the feature cannot silently produce wrong
/// activations.
#[cfg(feature = "triton-live")]
#[derive(Debug)]
pub struct TritonLocal {
    /// Model name inside the ONNX repository.
    pub model_name: String,
    /// Model version; `-1` means "latest".
    pub model_version: i64,
}

#[cfg(feature = "triton-live")]
impl ActivationModel for TritonLocal {
    fn input_shape(&self) -> InputShape {
        InputShape::Frames(0)
    }

    fn predict(&mut self, _features: &ArrayD<f32>) -> Result<ArrayD<f32>, InferenceError> {
        Err(InferenceError::NotImplemented(
            "triton in-process inference (task 2.3)",
        ))
    }
}
