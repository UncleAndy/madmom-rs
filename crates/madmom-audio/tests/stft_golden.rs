//! STFT against the reference dump (`tests/golden/sample.stft.npy`).
//!
//! Two tests, both about the same trap. `sample.stft.npy` holds what madmom produced
//! for `sample.wav`: `(281, 1024)` `complex64`, computed from int16 frames multiplied
//! by a Hann window scaled by `1/32767`. The first test requires agreement within the
//! plan's tolerance; the second shows how loud the failure is when the scaling is
//! skipped — the transform is off by a factor of 32767, which no detector would
//! report as an error.

use madmom_audio::signal::{
    DType, EndOfSignal, FramedSignal, Origin, Signal, FRAME_SIZE, HOP_SIZE,
};
use madmom_audio::stft::{stft, ShortTimeFourierTransformProcessor, WindowFunction};
use madmom_testkit::{load_golden, SPEC_ATOL};
use num_complex::Complex32;

fn max_abs_diff(actual: &ndarray::Array2<Complex32>, expected: &ndarray::Array2<Complex32>) -> f32 {
    assert_eq!(actual.shape(), expected.shape(), "shapes differ");
    actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0f32, f32::max)
}

fn reference_frames() -> Option<FramedSignal> {
    let signal = load_golden("sample.signal.npy")?;
    let samples = signal
        .into_dimensionality::<ndarray::Ix1>()
        .expect("the reference signal is mono")
        .to_owned();
    let signal = Signal::mono(samples, DType::Int16, Some(44_100));
    Some(FramedSignal::new(
        signal,
        FRAME_SIZE,
        HOP_SIZE,
        Origin::Center,
        EndOfSignal::Normal,
    ))
}

fn reference_stft() -> Option<ndarray::Array2<Complex32>> {
    let array = madmom_io::npy::read_npy(madmom_testkit::golden("sample.stft.npy")).ok()?;
    let values = array.as_complex32().expect("stft dump is complex64");
    let shape: Vec<usize> = array.shape().to_vec();
    Some(
        ndarray::Array2::from_shape_vec((shape[0], shape[1]), values)
            .expect("dump shape must match its data"),
    )
}

#[test]
fn stft_matches_the_reference_within_tolerance() {
    let Some(frames) = reference_frames() else {
        return;
    };
    let Some(expected) = reference_stft() else {
        return;
    };

    let transform = ShortTimeFourierTransformProcessor::default()
        .process(frames)
        .expect("the default processor must run");
    assert_eq!(transform.num_frames(), expected.shape()[0]);
    assert_eq!(transform.num_bins(), expected.shape()[1]);
    assert_eq!(transform.fft_size(), FRAME_SIZE);

    let diff = max_abs_diff(&transform.values().to_owned(), &expected);
    eprintln!("largest |stft - reference| = {diff:.3e} (tolerance {SPEC_ATOL:.0e})");
    assert!(
        diff <= SPEC_ATOL,
        "stft must agree within {SPEC_ATOL}, largest difference {diff}"
    );
}

#[test]
fn skipping_the_window_scaling_is_wrong_by_the_dtype_range() {
    let Some(frames) = reference_frames() else {
        return;
    };
    let Some(expected) = reference_stft() else {
        return;
    };

    // the trap: the very same Hann window, but *not* divided by 32767
    let unscaled = WindowFunction::Hanning.build(FRAME_SIZE);
    let as_f64 = frames.to_array().mapv(|value| value as f64);
    let wrong =
        stft(as_f64.view(), Some(unscaled.view()), None, false, false).expect("stft must run");

    let diff = max_abs_diff(&wrong, &expected);
    eprintln!("largest |unscaled stft - reference| = {diff:.3e}");
    assert!(
        diff > 1.0,
        "forgetting the 1/iinfo(dtype).max window scaling must be visible, got {diff}"
    );
}
