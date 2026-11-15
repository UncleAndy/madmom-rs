//! Tempo estimation against the reference dumps.
//!
//! The dumps come from madmom's own processors over its own activation fixtures
//! (`tools/dump_tempo.py`), so every stage is pinned without the network: the comb and
//! autocorrelation histograms, their smoothing, the detected tempi of the comb/acf/DBN
//! sources, and the TCN tempo layer. Tolerances are the plan's `f64` DSP bounds
//! (1e-6 absolute, 1e-9 relative).

use std::path::{Path, PathBuf};

use madmom_features::tempo::{
    detect_tempo, dominant_interval, interval_histogram_acf, interval_histogram_comb,
    smooth_histogram, ACFTempoHistogramProcessor, CombFilterTempoHistogramProcessor,
    DBNTempoHistogramProcessor, Histogram, TCNTempoHistogramProcessor, TempoError,
    TempoEstimationProcessor, TempoHistogramSource,
};
use madmom_testkit::load_activations;
use ndarray::ArrayView1;

const DSP_ATOL: f64 = 1e-6;
const DSP_RTOL: f64 = 1e-9;

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

/// Read one dump, or `None` when it is missing and the run does not require goldens.
fn load(tag: &str) -> Option<Vec<f64>> {
    let path = golden_dir().join("tempo").join(format!("{tag}.npy"));
    if !path.is_file() {
        assert!(
            std::env::var("MADMOM_REQUIRE_GOLDEN").is_err(),
            "{} is missing: run tools/dump_tempo.py first (see tests/golden/README.md)",
            path.display()
        );
        return None;
    }
    let array = madmom_io::npy::read_npy(&path).expect("dump must be readable");
    Some(array.as_f64().expect("tempo dumps hold doubles"))
}

/// Compare against a dump, reporting the largest difference seen.
fn assert_close(got: &[f64], want: &[f64], label: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: length {} against {}",
        got.len(),
        want.len()
    );
    let mut worst = 0.0f64;
    let mut worst_index = 0usize;
    for (index, (got, want)) in got.iter().zip(want.iter()).enumerate() {
        let bound = DSP_ATOL + DSP_RTOL * want.abs();
        let diff = (got - want).abs();
        if diff > bound {
            panic!("{label}: element {index} is {got} against {want} (bound {bound})");
        }
        if diff > worst {
            worst = diff;
            worst_index = index;
        }
    }
    eprintln!(
        "{label}: {} values, largest difference {worst:.3e} at {worst_index}",
        got.len()
    );
}

/// The beat activation function the reference's own tests use.
fn beat_activations() -> Vec<f64> {
    let (activations, fps) = load_activations("activations/sample.beats_blstm.npz")
        .expect("the beat activations fixture must be readable");
    assert_eq!(activations.shape(), &[281]);
    assert_eq!(fps, Some(100.0));
    activations.iter().map(|value| *value as f64).collect()
}

#[test]
fn tempo_functions_match_the_reference() {
    let act = beat_activations();
    let Some(expected_act) = load("act") else {
        return;
    };
    assert_close(&act, &expected_act, "act");

    let comb = interval_histogram_comb(ArrayView1::from(&act), 0.79, 24, Some(150))
        .expect("comb histogram");
    let Some(bins) = load("hist_comb_bins") else {
        return;
    };
    assert_close(comb.bins().to_vec().as_slice(), &bins, "hist_comb_bins");
    assert_close(
        comb.delays().to_vec().as_slice(),
        &load("hist_comb_delays").expect("delays dump"),
        "hist_comb_delays",
    );

    let acf = interval_histogram_acf(ArrayView1::from(&act), 24, Some(150)).expect("acf histogram");
    assert_close(
        acf.bins().to_vec().as_slice(),
        &load("hist_acf_bins").expect("acf dump"),
        "hist_acf_bins",
    );

    let smoothed = smooth_histogram(&comb, 3).expect("smoothing");
    assert_close(
        smoothed.bins().to_vec().as_slice(),
        &load("hist_comb_bins_smooth3").expect("smoothed dump"),
        "hist_comb_bins_smooth3",
    );

    let dominant = dominant_interval(&comb, None).expect("dominant interval");
    assert_eq!(dominant, 34.0, "the reference's dominant interval");
    assert_close(
        &[dominant],
        &load("dominant_interval").expect("dominant dump"),
        "dominant_interval",
    );

    let with_fps = detect_tempo(&comb, Some(100.0), false).expect("detect_tempo with fps");
    let expected = load("detect_tempo_hist_bpm").expect("bpm dump");
    assert_eq!(with_fps.shape()[1], 2);
    assert_close(
        with_fps.iter().copied().collect::<Vec<_>>().as_slice(),
        &expected,
        "detect_tempo_hist_bpm",
    );
    let delays = detect_tempo(&comb, None, false).expect("detect_tempo without fps");
    assert_close(
        delays.iter().copied().collect::<Vec<_>>().as_slice(),
        &load("detect_tempo_hist_delays").expect("delays dump"),
        "detect_tempo_hist_delays",
    );

    // the activations are smoothed over int(round(100 * 0.14)) = 14 frames
    let smoothed_act =
        madmom_audio::signal::smooth_f64(ArrayView1::from(&act), 14).expect("smooth");
    assert_close(
        smoothed_act.to_vec().as_slice(),
        &load("act_smoothed").expect("smoothed activations dump"),
        "act_smoothed",
    );
}

#[test]
fn comb_filter_tempo_estimation_matches_the_reference() {
    let act = beat_activations();
    let processor = CombFilterTempoHistogramProcessor::with_defaults(100.0);
    assert_eq!(processor.min_bpm(), 40.0);
    assert_eq!(processor.max_bpm(), 250.0);
    assert_eq!(processor.alpha(), 0.79);
    assert_eq!(processor.min_interval(), 24);
    assert_eq!(processor.max_interval(), 150);

    let histogram = processor
        .histogram(ArrayView1::from(&act))
        .expect("histogram");
    let Some(bins) = load("comb_hist_bins") else {
        return;
    };
    assert_close(
        histogram.bins().to_vec().as_slice(),
        &bins,
        "comb_hist_bins",
    );
    let smoothed = smooth_histogram(&histogram, 9).expect("smoothing");
    assert_close(
        smoothed.bins().to_vec().as_slice(),
        &load("comb_hist_bins_smooth9").expect("smoothed dump"),
        "comb_hist_bins_smooth9",
    );

    let estimation = TempoEstimationProcessor::with_defaults(processor, 100.0);
    let tempi = estimation.process(ArrayView1::from(&act)).expect("tempi");
    let expected = load("tempi_comb").expect("tempi dump");
    assert_eq!(tempi.shape(), &[expected.len() / 2, 2]);
    assert_close(
        tempi.iter().copied().collect::<Vec<_>>().as_slice(),
        &expected,
        "tempi_comb",
    );
    // the reference's own test values, restated as a literal so a silent dump change shows
    assert_eq!(tempi[[0, 0]], 176.47058823529412);
    assert!(
        (tempi[[0, 1]] - 0.4747).abs() < 0.001,
        "strength {}",
        tempi[[0, 1]]
    );
}

#[test]
fn acf_and_dbn_tempo_estimation_match_the_reference() {
    let act = beat_activations();

    let acf_processor = ACFTempoHistogramProcessor::with_defaults(100.0);
    let acf_histogram = acf_processor
        .histogram(ArrayView1::from(&act))
        .expect("acf histogram");
    let Some(expected) = load("acf_hist_bins") else {
        return;
    };
    assert_close(
        acf_histogram.bins().to_vec().as_slice(),
        &expected,
        "acf_hist_bins",
    );
    let acf_tempi = TempoEstimationProcessor::with_defaults(acf_processor, 100.0)
        .process(ArrayView1::from(&act))
        .expect("acf tempi");
    assert_close(
        acf_tempi.iter().copied().collect::<Vec<_>>().as_slice(),
        &load("tempi_acf").expect("acf tempi dump"),
        "tempi_acf",
    );

    let dbn_processor = DBNTempoHistogramProcessor::with_defaults(100.0);
    let dbn_histogram = dbn_processor
        .histogram(ArrayView1::from(&act))
        .expect("dbn histogram");
    assert_close(
        dbn_histogram.bins().to_vec().as_slice(),
        &load("dbn_hist_bins").expect("dbn histogram dump"),
        "dbn_hist_bins",
    );
    // the reference's own assertions: a count histogram over 281 frames
    let total: f64 = dbn_histogram.bins().sum();
    assert_eq!(total, 281.0, "one interval per frame");
    let dbn_tempi = TempoEstimationProcessor::new(dbn_processor, None, 9, Some(100.0), false)
        .process(ArrayView1::from(&act))
        .expect("dbn tempi");
    assert_close(
        dbn_tempi.iter().copied().collect::<Vec<_>>().as_slice(),
        &load("tempi_dbn").expect("dbn tempi dump"),
        "tempi_dbn",
    );
    assert_eq!(dbn_tempi[[0, 0]], 176.47058823529412);
}

#[test]
fn tcn_tempo_layer_matches_the_reference() {
    let (activations, _) = load_activations("activations/sample.beats_tcn_tempo.npz")
        .expect("the TCN fixture must be readable");
    let data: Vec<f64> = activations.iter().map(|value| *value as f64).collect();
    let processor = TCNTempoHistogramProcessor::new(10.0, 250.0);
    let histogram = processor
        .histogram(ArrayView1::from(&data))
        .expect("tcn histogram");
    let Some(bins) = load("tcn_hist_bins") else {
        return;
    };
    assert_close(histogram.bins().to_vec().as_slice(), &bins, "tcn_hist_bins");
    assert_close(
        histogram.delays().to_vec().as_slice(),
        &load("tcn_hist_delays").expect("tcn delays dump"),
        "tcn_hist_delays",
    );
    assert_eq!(histogram.delays()[0], 10.0);
    assert_eq!(histogram.delays()[240], 250.0);

    let tempi = TempoEstimationProcessor::new(processor.clone(), None, 9, None, false)
        .process(ArrayView1::from(&data))
        .expect("tcn tempi");
    let expected = load("tcn_tempi").expect("tcn tempi dump");
    assert_eq!(tempi.shape(), &[expected.len() / 2, 2]);
    assert_close(
        tempi.iter().copied().collect::<Vec<_>>().as_slice(),
        &expected,
        "tcn_tempi",
    );
    // the reference's own test values
    assert_eq!(tempi[[0, 0]], 87.0);
    assert_eq!(tempi[[1, 0]], 174.0);

    let smoothed = TempoEstimationProcessor::new(processor, None, 15, None, false)
        .process(ArrayView1::from(&data))
        .expect("tcn tempi with a wider smoothing");
    assert_close(
        smoothed.iter().copied().collect::<Vec<_>>().as_slice(),
        &load("tcn_tempi_smooth15").expect("smoothed tcn tempi dump"),
        "tcn_tempi_smooth15",
    );
}

#[test]
fn interpolation_is_refused_rather_than_approximated() {
    let act = beat_activations();
    let comb = interval_histogram_comb(ArrayView1::from(&act), 0.79, 24, Some(150))
        .expect("comb histogram");
    // scipy's quadratic B-spline interpolation is not ported; the reference dump exists
    // (detect_tempo_hist_interp.npy, tempi_comb_interp.npy) so the gap is measurable
    assert!(
        load("detect_tempo_hist_interp").is_some()
            || std::env::var("MADMOM_REQUIRE_GOLDEN").is_err()
    );
    assert_eq!(
        detect_tempo(&comb, Some(100.0), true),
        Err(TempoError::InterpolationNotPorted)
    );
    let processor = CombFilterTempoHistogramProcessor::with_defaults(100.0);
    let estimation = TempoEstimationProcessor::new(processor, Some(0.14), 9, Some(100.0), true);
    assert_eq!(
        estimation.process(ArrayView1::from(&act)),
        Err(TempoError::InterpolationNotPorted)
    );
    let _ = Histogram::new(ndarray::array![1.0], ndarray::array![1.0])
        .expect("a one-bin histogram is valid");
}
