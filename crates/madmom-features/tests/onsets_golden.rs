//! Onset detection functions and peak picking against the reference dumps.
//!
//! The inputs are checked first (the raw spectrogram, its logarithmic filtered version and
//! the phase), then every ODF, then the peak picker and the processor on madmom's own
//! activation fixtures — including the online mode, which must report the same onsets as
//! the offline one when the activations are fed frame by frame.

use std::path::{Path, PathBuf};

use madmom_audio::signal::{DType, EndOfSignal, FramedSignal, Origin, Signal};
use madmom_audio::spectrogram::{
    FilterbankSettings, FilteredSpectrogramProcessor, LogarithmicFilteredSpectrogramProcessor,
    LogarithmicSpectrogramProcessor, Spectrogram, SpectrogramProcessor,
};
use madmom_audio::stft::{phase, ShortTimeFourierTransformProcessor};
use madmom_features::onsets::{
    complex_domain, complex_flux, high_frequency_content, modified_kullback_leibler,
    normalized_weighted_phase_deviation, peak_picking, phase_deviation, rectified_complex_domain,
    spectral_diff, spectral_flux, superflux, weighted_phase_deviation, OnsetPeakPickingProcessor,
};
use madmom_testkit::{fixture, load_activations};
use ndarray::{ArrayView1, Ix1};

/// The reference's ODF tolerances: the ODFs are float32 in madmom (the spectrogram is),
/// the port accumulates in float64, so the plan's spectral bound applies.
const ODF_ATOL: f64 = 1e-5;
const ODF_RTOL: f64 = 1e-5;
/// The input spectrogram and phase bounds.
const SPEC_ATOL: f64 = 1e-5;
/// The reference's phase is float32; the port evaluates it in float64.
const PHASE_TOLERANCE: f64 = 1e-6;

const FRAME_SIZE: usize = 2048;
const HOP_SIZE: f64 = 441.0;
const NUM_BANDS: usize = 24;

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

fn load(tag: &str) -> Option<Vec<f64>> {
    let path = golden_dir().join("onsets").join(format!("{tag}.npy"));
    if !path.is_file() {
        assert!(
            std::env::var("MADMOM_REQUIRE_GOLDEN").is_err(),
            "{} is missing: run tools/dump_onsets.py first (see tests/golden/README.md)",
            path.display()
        );
        return None;
    }
    let array = madmom_io::npy::read_npy(&path).expect("dump must be readable");
    Some(array.as_f64().expect("onset dumps hold doubles"))
}

fn assert_close(got: &[f64], want: &[f64], atol: f64, rtol: f64, label: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: length {} against {}",
        got.len(),
        want.len()
    );
    let mut worst = 0.0f64;
    for (index, (got, want)) in got.iter().zip(want.iter()).enumerate() {
        let bound = atol + rtol * want.abs();
        let diff = (got - want).abs();
        if diff > bound {
            panic!("{label}: element {index} is {got} against {want} (bound {bound})");
        }
        if diff > worst {
            worst = diff;
        }
    }
    eprintln!(
        "{label}: {} values, largest difference {worst:.3e}",
        got.len()
    );
}

/// The spectrogram chain the reference's own onset tests use.
fn spectrograms() -> (Spectrogram, Spectrogram, ndarray::Array2<f64>) {
    let audio = madmom_io::audio::read_wave(fixture("audio/sample.wav"))
        .expect("the reference sample must be readable");
    let samples = audio
        .samples
        .into_dimensionality::<Ix1>()
        .expect("the reference sample is mono");
    let signal = Signal::mono(samples, DType::Int16, Some(audio.sample_rate));
    let frames = FramedSignal::new(
        signal,
        FRAME_SIZE,
        HOP_SIZE,
        Origin::Center,
        EndOfSignal::Normal,
    );
    let transform = ShortTimeFourierTransformProcessor {
        circular_shift: true,
        ..Default::default()
    }
    .process(frames)
    .expect("the reference uses circular_shift=True for the onset ODFs");
    let phase = phase(transform.values());
    let spectrogram = SpectrogramProcessor.process(&transform);
    let log_filtered = LogarithmicFilteredSpectrogramProcessor {
        filter: FilteredSpectrogramProcessor {
            settings: FilterbankSettings {
                num_bands: NUM_BANDS,
                ..Default::default()
            },
        },
        log: LogarithmicSpectrogramProcessor::default(),
    }
    .process(&spectrogram)
    .expect("filtering must succeed");
    (spectrogram, log_filtered, phase)
}

#[test]
fn onset_detection_functions_match_the_reference() {
    let (spectrogram, log_filtered, phase) = spectrograms();

    let Some(expected) = load("spec") else {
        return;
    };
    assert_eq!(spectrogram.data().len(), expected.len());
    assert_close(
        &spectrogram
            .data()
            .iter()
            .map(|value| *value as f64)
            .collect::<Vec<_>>(),
        &expected,
        SPEC_ATOL,
        ODF_RTOL,
        "spec",
    );
    let filtered = load("log_filt_spec").expect("filtered dump");
    assert_close(
        &log_filtered
            .data()
            .iter()
            .map(|value| *value as f64)
            .collect::<Vec<_>>(),
        &filtered,
        SPEC_ATOL,
        ODF_RTOL,
        "log_filt_spec",
    );
    let expected_phase = load("phase").expect("phase dump");
    // madmom's phase is float32 (`np.angle` of a complex64 transform) while the port
    // evaluates atan2 in float64, and a value of exactly +-pi flips sign between the two.
    // The wrapped difference is the meaningful comparison; the ODFs below are unaffected,
    // because e^{i*pi} == e^{-i*pi}.
    let mut worst_phase = 0.0f64;
    for (got, want) in phase.iter().zip(expected_phase.iter()) {
        let difference = (got - want + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
            - std::f64::consts::PI;
        if difference.abs() > worst_phase {
            worst_phase = difference.abs();
        }
    }
    assert!(
        worst_phase < PHASE_TOLERANCE,
        "phase: largest wrapped difference {worst_phase:.3e} against the float32 bound {PHASE_TOLERANCE:.0e}"
    );
    eprintln!(
        "phase: {} values, largest wrapped difference {worst_phase:.3e}",
        phase.len()
    );

    // the magnitude-only ODFs run on the logarithmic filtered spectrogram
    let filtered = [
        ("odf_hfc", high_frequency_content(&log_filtered)),
        (
            "odf_spectral_diff",
            spectral_diff(&log_filtered, None).expect("spectral diff"),
        ),
        (
            "odf_spectral_flux",
            spectral_flux(&log_filtered, None).expect("spectral flux"),
        ),
        (
            "odf_superflux",
            superflux(&log_filtered, None, 3).expect("superflux"),
        ),
        (
            "odf_complex_flux",
            complex_flux(&log_filtered, phase.view(), None, 3, 3, 0).expect("complex flux"),
        ),
        (
            "odf_mkl",
            modified_kullback_leibler(&log_filtered, 1, madmom_features::onsets::EPSILON)
                .expect("mkl"),
        ),
        (
            "odf_phase_deviation",
            phase_deviation(&log_filtered, phase.view()).expect("phase deviation"),
        ),
    ];
    for (tag, odf) in filtered {
        let Some(expected) = load(tag) else {
            return;
        };
        assert_close(odf.to_vec().as_slice(), &expected, ODF_ATOL, ODF_RTOL, tag);
    }

    // the phase-weighted ones need the unfiltered spectrogram (the reference refuses a
    // filtered one, and so does the port)
    let raw = [
        (
            "odf_weighted_phase_deviation",
            weighted_phase_deviation(&spectrogram, phase.view()).expect("wpd"),
        ),
        (
            "odf_normalized_wpd",
            normalized_weighted_phase_deviation(
                &spectrogram,
                phase.view(),
                madmom_features::onsets::EPSILON,
            )
            .expect("normalized wpd"),
        ),
        (
            "odf_complex_domain",
            complex_domain(&spectrogram, phase.view()).expect("complex domain"),
        ),
        (
            "odf_rectified_complex_domain",
            rectified_complex_domain(&spectrogram, phase.view(), None)
                .expect("rectified complex domain"),
        ),
    ];
    for (tag, odf) in raw {
        let Some(expected) = load(tag) else {
            return;
        };
        assert_close(odf.to_vec().as_slice(), &expected, ODF_ATOL, ODF_RTOL, tag);
    }

    // and they must refuse a filtered spectrogram, exactly like the reference
    assert!(weighted_phase_deviation(&log_filtered, phase.view()).is_err());
    assert!(complex_domain(&log_filtered, phase.view()).is_err());
}

/// An activation fixture and its frame rate. The fixtures do **not** all run at 100 fps:
/// the SuperFlux one is at 200 fps, which is why its first onset is frame 2 at 0.01 s.
fn activations(tag: &str) -> (Vec<f64>, f64) {
    let (activations, fps) = load_activations(tag).expect("the fixture must be readable");
    let fps = fps.expect("the fixtures carry their frame rate");
    (activations.iter().map(|value| *value as f64).collect(), fps)
}

#[test]
fn peak_picking_matches_the_reference_on_its_fixtures() {
    let (superflux_activations, superflux_fps) = activations("activations/sample.super_flux.npz");
    assert_eq!(
        superflux_fps, 200.0,
        "the SuperFlux fixture runs at 200 fps"
    );
    // the reference's defaults are pre_max=1 and post_max=1
    let cases: [(&str, Vec<usize>); 3] = [
        (
            "pp_superflux_default",
            peak_picking(
                ArrayView1::from(&superflux_activations),
                1.1,
                None,
                0,
                0,
                1,
                1,
            )
            .expect("peak picking"),
        ),
        (
            "pp_superflux_smooth3",
            peak_picking(
                ArrayView1::from(&superflux_activations),
                1.1,
                Some(3),
                0,
                0,
                1,
                1,
            )
            .expect("peak picking with smoothing"),
        ),
        (
            "pp_superflux_params",
            peak_picking(
                ArrayView1::from(&superflux_activations),
                1.1,
                None,
                30,
                0,
                2,
                10,
            )
            .expect("peak picking with a moving average"),
        ),
    ];
    for (tag, frames) in cases {
        let Some(expected) = load(tag) else {
            return;
        };
        let expected: Vec<f64> = expected;
        assert_eq!(
            frames.iter().map(|frame| *frame as f64).collect::<Vec<_>>(),
            expected,
            "{tag}: the peak frames must match exactly"
        );
        eprintln!("{tag}: {} peaks identical", frames.len());
    }

    let (rnn_activations, rnn_fps) = activations("activations/sample.onsets_rnn.npz");
    assert_eq!(rnn_fps, 100.0);
    let Some(expected) = load("pp_rnn_post_max0") else {
        return;
    };
    let frames = peak_picking(ArrayView1::from(&rnn_activations), 0.23, None, 0, 0, 1, 0)
        .expect("peak picking without a post maximum");
    assert_eq!(
        frames.iter().map(|frame| *frame as f64).collect::<Vec<_>>(),
        expected,
        "pp_rnn_post_max0"
    );
}

#[test]
fn onset_peak_picking_processor_matches_the_reference() {
    let (superflux_activations, superflux_fps) = activations("activations/sample.super_flux.npz");
    let processor = OnsetPeakPickingProcessor::new(
        1.1,
        0.0,
        0.15,
        0.0,
        0.01,
        0.05,
        0.03,
        0.0,
        superflux_fps,
        false,
    );
    let onsets = processor
        .process_offline(ArrayView1::from(&superflux_activations))
        .expect("offline onsets");
    let Some(expected) = load("onsets_offline") else {
        return;
    };
    assert_eq!(onsets.len(), expected.len(), "onset count: {onsets:?}");
    assert_close(&onsets, &expected, 1e-6, 0.0, "onsets_offline");

    // online: the same onsets, whether the activations arrive at once or frame by frame
    let (rnn_activations, rnn_fps) = activations("activations/sample.onsets_rnn.npz");
    // the reference's defaults: combine 0.03 (the online constructor only forces smooth,
    // post_avg and post_max to zero)
    let mut online =
        OnsetPeakPickingProcessor::new(0.23, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.0, rnn_fps, true);
    assert_eq!(online.threshold(), 0.23);
    let all_at_once = online
        .process_online(&rnn_activations, false)
        .expect("online onsets");
    let Some(expected) = load("onsets_online_all") else {
        return;
    };
    assert_close(&all_at_once, &expected, 1e-6, 0.0, "onsets_online_all");

    online.reset();
    let mut frame_by_frame: Vec<f64> = Vec::new();
    for activation in &rnn_activations {
        let onsets = online
            .process_online(&[*activation], false)
            .expect("online onsets, frame by frame");
        frame_by_frame.extend(onsets);
    }
    let expected_frames = load("onsets_online_frames").expect("frame dump");
    assert_close(
        &frame_by_frame,
        &expected_frames,
        1e-6,
        0.0,
        "onsets_online_frames",
    );
}
