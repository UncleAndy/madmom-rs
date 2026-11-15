//! Framing against the reference dump (`tests/golden/`, see its README).
//!
//! This is the strongest test of the framing rule: 281 frames × 2048 samples produced
//! by madmom for a real file, compared bit for bit. A tolerance would hide the two
//! traps this stage has (centred frames, `ceil(num_samples / hop_size)` frames): an
//! off-by-1024 shift or a dropped trailing frame changes nearly every sample, and both
//! would still "look plausible" in a downstream detector.

use madmom_audio::signal::{
    DType, EndOfSignal, FramedSignal, Origin, Signal, FRAME_SIZE, HOP_SIZE,
};
use madmom_testkit::{load_golden, max_abs_diff};

/// Number of samples in the reference file `sample.wav`.
const SAMPLE_WAV_SAMPLES: usize = 123_481;

#[test]
fn framing_of_the_reference_sample_is_bit_exact() {
    let Some(signal) = load_golden("sample.signal.npy") else {
        return;
    };
    let Some(expected) = load_golden("sample.frames.npy") else {
        return;
    };

    let samples = signal
        .into_dimensionality::<ndarray::Ix1>()
        .expect("the reference signal is mono")
        .to_owned();
    assert_eq!(samples.len(), SAMPLE_WAV_SAMPLES, "unexpected sample count");

    let signal = Signal::mono(samples, DType::Int16, Some(44_100));
    let framed = FramedSignal::new(
        signal,
        FRAME_SIZE,
        HOP_SIZE,
        Origin::Center,
        EndOfSignal::Normal,
    );

    // 281 = ceil(123481 / 441); the naive "full frames only" rule gives 276
    assert_eq!(framed.num_frames(), expected.shape()[0]);
    assert_eq!(framed.frame_size(), expected.shape()[1]);
    assert_eq!(framed.frame_rate(), Some(100.0));

    let frames = framed.to_array().into_dyn();
    let diff = max_abs_diff(&frames, &expected);
    assert_eq!(
        diff, 0.0,
        "framing must be bit-exact, largest difference {diff}"
    );
}
