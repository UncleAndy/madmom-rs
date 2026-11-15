//! Music information retrieval features: activations, onsets, beats, beats_hmm, beats_crf, downbeats, tempo, notes, chords, key.
//!
//! Beats first (the flagship), the remaining families follow in their phases.

#![deny(missing_docs)]

pub mod beats;
pub mod beats_hmm;
pub mod downbeats;
