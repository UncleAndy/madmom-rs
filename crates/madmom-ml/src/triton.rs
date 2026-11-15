//! In-process Triton Inference Server backend (feature `triton-live`).
//!
//! Runs the ONNX models converted from madmom networks inside this process, through the
//! C-API of `libtritonserver.so` (built by `tools/build_triton_core.sh` and
//! `tools/build_onnx_backend.sh`), and implements [`ActivationModel`] with it.
//!
//! The bindings are asynchronous and take a handle on a tokio runtime when the server
//! starts, so the model owns a runtime and every call blocks on it. One server is started
//! per [`TritonModel`] and stopped
//! when it is dropped.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use ndarray::{ArrayD, IxDyn};
use tokio::runtime::Runtime;
use tritonserver_rs::memory::Buffer;
use tritonserver_rs::options::{Control, Options};
use tritonserver_rs::server::Server;

use crate::inference::{ActivationModel, InferenceError, InputShape};

/// How the inference server should be started.
#[derive(Debug, Clone)]
pub struct TritonConfig {
    /// Directory holding one subdirectory per model (`<name>/<version>/model.onnx`).
    pub repository: PathBuf,
    /// Where the backends live; the C-API defaults to `/opt/tritonserver/backends`.
    pub backend_directory: Option<PathBuf>,
    /// Model version to request, or -1 for the latest.
    pub model_version: i64,
    /// Model to load eagerly at startup, if any.
    pub startup_model: Option<String>,
    /// How long the server waits for in-flight requests when stopping.
    pub exit_timeout: Duration,
}

impl TritonConfig {
    /// Configuration for a repository, with the defaults the plan fixed: explicit model
    /// control, no startup model, a five second exit timeout.
    pub fn new<P: Into<PathBuf>>(repository: P) -> Self {
        Self {
            repository: repository.into(),
            backend_directory: None,
            model_version: -1,
            startup_model: None,
            exit_timeout: Duration::from_secs(5),
        }
    }

    /// Point the server at a backend directory (the built `libtriton_onnxruntime.so`).
    pub fn with_backend_directory<P: Into<PathBuf>>(mut self, dir: P) -> Self {
        self.backend_directory = Some(dir.into());
        self
    }
}

/// A model served by the in-process server.
pub struct TritonModel {
    server: Arc<Server>,
    /// The bindings keep a handle on this runtime, so it has to outlive the server.
    runtime: Runtime,
    model_name: String,
    model_version: i64,
    input_name: String,
    output_name: String,
    shape: InputShape,
    fps: Option<f64>,
}

impl TritonModel {
    /// Starts a server for `config` and prepares `model_name` for inference.
    ///
    /// `input_name` and `output_name` are the tensor names declared in the model`+"`"+`s
    /// `config.pbtxt`; `shape` is the shape of the feature matrix the model consumes.
    pub fn start(
        config: &TritonConfig,
        model_name: &str,
        input_name: &str,
        output_name: &str,
        shape: InputShape,
    ) -> Result<Self, InferenceError> {
        let mut options =
            Options::new(&config.repository).map_err(|e| InferenceError::triton(model_name, e))?;
        let builder = options
            .model_control_mode(Control::Explicit)
            .map_err(|e| InferenceError::triton(model_name, e))?;
        if let Some(dir) = &config.backend_directory {
            builder
                .backend_directory(dir)
                .map_err(|e| InferenceError::triton(model_name, e))?;
        }
        if let Some(model) = &config.startup_model {
            builder
                .startup_model(model)
                .map_err(|e| InferenceError::triton(model_name, e))?;
        }
        builder
            .exit_timeout(config.exit_timeout)
            .and_then(|opts| opts.pinned_memory_pool_byte_size(0))
            .map_err(|e| InferenceError::triton(model_name, e))?;

        let runtime = Runtime::new().map_err(|e| InferenceError::triton(model_name, e))?;
        let mut server = runtime
            .block_on(Server::new(options))
            .map_err(|e| InferenceError::triton(model_name, e))?;
        // Explicit model control means the server loads nothing on startup, and loading is
        // asynchronous: ask for the model, then wait until it is ready to serve.
        server
            .load_model(model_name)
            .map_err(|e| InferenceError::triton(model_name, e))?;
        wait_until_ready(&server, model_name, config.model_version)?;
        Ok(Self {
            server: Arc::new(server),
            runtime,
            model_name: model_name.to_owned(),
            model_version: config.model_version,
            input_name: input_name.to_owned(),
            output_name: output_name.to_owned(),
            shape,
            fps: None,
        })
    }

    /// Frame rate reported for the activations, when set by the caller.
    pub fn with_fps(mut self, fps: f64) -> Self {
        self.fps = Some(fps);
        self
    }

    /// The model repository this server was started for.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

impl ActivationModel for TritonModel {
    fn input_shape(&self) -> InputShape {
        self.shape
    }

    fn predict(&mut self, features: &ArrayD<f32>) -> Result<ArrayD<f32>, InferenceError> {
        let frames = features.shape()[0];
        let dims = self.dims(frames, features.shape())?;
        let flat: Vec<f32> = features.iter().copied().collect();
        let model = self.model_name.clone();

        let mut request = self
            .server
            .create_request(&model, self.model_version)
            .map_err(|e| InferenceError::triton(&model, e))?;
        request.add_default_allocator();
        request
            .add_input_with_dims(&self.input_name, Buffer::from(flat), &dims)
            .map_err(|e| InferenceError::triton(&model, e))?;
        let future = request
            .infer_async()
            .map_err(|e| InferenceError::triton(&model, e))?;
        let response = self
            .runtime
            .block_on(future)
            .map_err(|e| InferenceError::triton(&model, e))?;

        let output =
            response
                .get_output(&self.output_name)
                .ok_or_else(|| InferenceError::Triton {
                    model: model.clone(),
                    cause: format!("response carries no output {}", self.output_name),
                })?;
        let owned = output
            .get_buffer()
            .get_owned_slice(..)
            .map_err(|e| InferenceError::triton(&model, e))?;
        let (prefix, values, suffix) = unsafe { owned.align_to::<f32>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            return Err(InferenceError::Triton {
                model,
                cause: "output buffer is not a whole number of f32 values".to_owned(),
            });
        }
        if frames == 0 || values.len() % frames != 0 {
            return Err(InferenceError::Triton {
                model,
                cause: format!(
                    "{} output values do not divide over {} frames",
                    values.len(),
                    frames
                ),
            });
        }
        let activations =
            ArrayD::from_shape_vec(IxDyn(&[frames, values.len() / frames]), values.to_vec())
                .expect("shape and data length agree by construction");
        Ok(activations)
    }

    fn fps(&self) -> Option<f64> {
        self.fps
    }
}

impl TritonModel {
    /// Frame-major dimensions for the model, with the frame count from `features`.
    fn dims(&self, frames: usize, shape: &[usize]) -> Result<Vec<i64>, InferenceError> {
        let expected = match self.shape {
            InputShape::Frames(n) => vec![frames, n],
            InputShape::FramesFreq(bins, n) => vec![frames, bins, n],
            InputShape::FramesFreqCh(bins, ch, n) => vec![frames, bins, ch, n],
        };
        if expected[1..] != shape[1..] {
            return Err(InferenceError::ShapeMismatch {
                expected: self.shape,
                got: shape.to_vec(),
            });
        }
        Ok(expected.into_iter().map(|d| d as i64).collect())
    }
}

impl Drop for TritonModel {
    fn drop(&mut self) {
        if let Err(e) = self.server.stop() {
            log_stop_failure(&e);
        }
    }
}

/// Waits for a loaded model to become ready, up to roughly a minute.
fn wait_until_ready(server: &Server, model_name: &str, version: i64) -> Result<(), InferenceError> {
    for _ in 0..600 {
        if server
            .model_is_ready(model_name, version)
            .map_err(|e| InferenceError::triton(model_name, e))?
        {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Err(InferenceError::Triton {
        model: model_name.to_owned(),
        cause: "model did not become ready within a minute".to_owned(),
    })
}

/// Stopping is best effort: the process is going away anyway, but a failure is worth seeing.
fn log_stop_failure<E: std::fmt::Display>(e: &E) {
    eprintln!("madmom-ml: stopping the Triton server failed: {e}");
}
