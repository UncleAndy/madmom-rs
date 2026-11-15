//! Audio front end: signals, framing, STFT, filterbanks, spectrograms, chroma.
//!
//! Phase 1 of the port. The modules mirror `madmom/audio/*.py` one to one so that
//! a failure can always be traced back to a single reference module:
//!
//! - [`signal`] — `madmom/audio/signal.py`: samples with metadata, framing, and the
//!   scalar signal functions (smoothing, gain, normalize, remix, rescale, trim,
//!   energy, RMS, sound pressure level).
//! - [`stft`] — `madmom/audio/stft.py`: the short-time Fourier transform, its windows
//!   and the bin frequencies.
//!
//! Numeric fidelity: samples are held as `f32` together with the [`signal::DType`]
//! they were loaded as, because madmom's DSP decisions (window scaling by
//! `iinfo(dtype).max`, integer casts in `normalize`/`adjust_gain`) depend on the
//! original format rather than on the values alone.

#![deny(missing_docs)]

pub mod signal;
pub mod stft;
