//! The spectrogram chain against the reference dumps.
//!
//! Replays exactly what `tools/dump_reference.py` dumped for `sample.wav`:
//!
//! ```text
//! signal → frames (2048/441) → STFT (default: Hann window, int16 scaling)
//!        → magnitudes → filterbank (logarithmic, 6 bands/octave, 30..17000, normalized)
//!        → log10(1 + x) → positive differences (ratio 0.5, stacked horizontally)
//! ```
//!
//! Each stage is compared with the dump of that stage, so a failure points at the stage
//! that broke instead of at "the beats do not match".

use std::path::{Path, PathBuf};

use madmom_audio::signal::{
    DType, EndOfSignal, FramedSignal, Origin, Signal, FRAME_SIZE, HOP_SIZE,
};
use madmom_audio::spectrogram::{
    DifferenceParameters, FilterbankSettings, FilteredSpectrogramProcessor,
    LogarithmicFilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor, Spectrogram,
    SpectrogramDifferenceProcessor, SpectrogramProcessor, StackDiffs,
};
use madmom_audio::stft::ShortTimeFourierTransformProcessor;
use madmom_testkit::{load_golden, SPEC_ATOL};

/// The filterbank settings of the dumped chain (`dump_reference.py --num-bands 6`).
const NUM_BANDS: usize = 6;
const FMIN: f64 = 30.0;
const FMAX: f64 = 17000.0;

fn golden_dir() -> PathBuf {
    std::env::var("MADMOM_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace_root().join("tests").join("golden"))
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate lives in <root>/crates/madmom-audio")
        .to_path_buf()
}

fn reference(golden: &str) -> Option<Vec<f32>> {
    let path = golden_dir().join(golden);
    if !path.is_file() {
        assert!(
            std::env::var("MADMOM_REQUIRE_GOLDEN").is_err(),
            "{} is missing: generate the dumps first (see tests/golden/README.md)",
            path.display()
        );
        return None;
    }
    let array = madmom_io::npy::read_npy(&path).expect("dump must be readable");
    Some(array.as_f32().expect("dumps hold numeric values"))
}

/// Compares one stage against its dump and reports the difference.
fn stage_diff(stage: &str, got: ndarray::ArrayView2<'_, f32>, golden: &str) -> f32 {
    let Some(expected) = reference(golden) else {
        return 0.0;
    };
    assert_eq!(
        got.len(),
        expected.len(),
        "{stage}: {} values against {} in {golden}",
        got.len(),
        expected.len()
    );
    let diff = got
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("{stage:<12} max |Δ| = {diff:.3e} (tolerance {SPEC_ATOL:.0e})");
    assert!(
        diff <= SPEC_ATOL,
        "{stage}: largest difference {diff:.3e} exceeds {SPEC_ATOL:.0e}"
    );
    diff
}

#[test]
fn spectrogram_chain_matches_the_reference_stage_by_stage() {
    // the signal dump is the anchor: without it there is nothing to compare
    let Some(signal) = load_golden("sample.signal.npy") else {
        return;
    };
    let samples = signal
        .into_dimensionality::<ndarray::Ix1>()
        .expect("the reference signal is mono")
        .to_owned();
    let signal = Signal::mono(samples, DType::Int16, Some(44_100));
    let frames = FramedSignal::new(
        signal,
        FRAME_SIZE,
        HOP_SIZE,
        Origin::Center,
        EndOfSignal::Normal,
    );
    let transform = ShortTimeFourierTransformProcessor::default()
        .process(frames)
        .expect("the default processor must run");

    let spectrogram: Spectrogram = SpectrogramProcessor.process(&transform);
    stage_diff("spec", spectrogram.data(), "sample.spec.npy");

    let filtered = FilteredSpectrogramProcessor {
        settings: FilterbankSettings {
            num_bands: NUM_BANDS,
            fmin: FMIN,
            fmax: FMAX,
            ..Default::default()
        },
    }
    .process(&spectrogram)
    .expect("filtering must succeed");
    stage_diff("filt", filtered.data(), "sample.filt.npy");

    let logged = LogarithmicSpectrogramProcessor::default().process(&filtered);
    stage_diff("logspec", logged.data(), "sample.logspec.npy");

    let difference = SpectrogramDifferenceProcessor {
        parameters: DifferenceParameters {
            diff_ratio: 0.5,
            positive_diffs: true,
            stack_diffs: StackDiffs::Horizontal,
            ..Default::default()
        },
    }
    .process(&logged)
    .expect("the difference must run");
    stage_diff("diff", difference.data(), "sample.diff.npy");

    // the number of difference frames the reference derives from the window shape
    assert_eq!(
        difference
            .difference_parameters()
            .and_then(|p| p.diff_frames),
        Some(1),
        "madmom's _diff_frames(0.5, 441, 2048) == 1"
    );

    // the combined processor (the one the beat chains use) must give the same logspec
    let combined = LogarithmicFilteredSpectrogramProcessor {
        filter: FilteredSpectrogramProcessor {
            settings: FilterbankSettings {
                num_bands: NUM_BANDS,
                fmin: FMIN,
                fmax: FMAX,
                ..Default::default()
            },
        },
        ..Default::default()
    }
    .process(&spectrogram)
    .expect("the combined processor must run");
    stage_diff("logspec(1x)", combined.data(), "sample.logspec.npy");
}
