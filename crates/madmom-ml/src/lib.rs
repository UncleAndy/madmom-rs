//! Machine learning glue: HMM (Viterbi/forward), inference sources (golden fixtures, in-process Triton), model repository registry.
//!
//! Phase 0 scaffold: only the module skeleton exists here.

#![deny(missing_docs)]

pub mod inference;

pub mod hmm;

/// In-process Triton Inference Server (feature `triton-live`).
#[cfg(feature = "triton-live")]
pub mod triton;

#[cfg(feature = "triton-live")]
pub use triton::{TritonConfig, TritonModel};
