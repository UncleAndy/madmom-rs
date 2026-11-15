//! Filterbanks and frequency tables against the reference dumps
//! (`tests/golden/filters/`, produced by `tools/dump_filters.py`).
//!
//! The case list is written out here instead of being derived from the index, so a
//! renamed or added dump fails loudly. Every case is built with the same bin
//! frequencies the dumps use (2048-point FFT at 44.1 kHz, 1024 bins).
//!
//! Note the two oddities of the Bark filterbank that the dumps pin down: it is built
//! from non-overlapping **rectangular** filters, and its `unique_filters` flag is
//! **inverted** relative to the other filterbanks.

use std::path::{Path, PathBuf};

use madmom_audio::filters::{
    bark_double_frequencies, bark_filterbank, bark_frequencies, erb2hz, hz2erb, hz2mel, hz2midi,
    logarithmic_filterbank, mel_filterbank, mel_frequencies, midi2hz, semitone_frequencies,
    BarkSpacing, Filterbank, A4,
};
use madmom_audio::stft::fft_frequencies;
use ndarray::{Array1, ArrayView1};

/// Absolute tolerance for the filterbank matrices.
///
/// The matrices themselves are built by integer/float arithmetic that is identical on
/// both sides; only the normalization sum differs, because NumPy sums a float32 buffer
/// pairwise while the port sums it linearly. The observed maximum difference is two
/// orders of magnitude below this bound (printed by the test).
const FILTER_ATOL: f32 = 1e-6;

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

fn bin_frequencies() -> Array1<f64> {
    fft_frequencies(1024, 44_100.0)
}

fn reference(dir: &Path, tag: &str) -> (Vec<f32>, Vec<usize>) {
    let path = dir.join(format!("{tag}.npy"));
    let array =
        madmom_io::npy::read_npy(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let shape = array.shape().to_vec();
    (array.as_f32().expect("dumps hold numeric values"), shape)
}

fn close_vec(got: &[f32], expected: &[f32], what: &str) -> f32 {
    assert_eq!(got.len(), expected.len(), "{what}: length differs");
    let diff = got
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff <= FILTER_ATOL,
        "{what}: largest difference {diff:.3e} exceeds {FILTER_ATOL:.0e}"
    );
    diff
}

fn close_filterbank(
    matrix: ndarray::ArrayView2<'_, f32>,
    expected: &[f32],
    shape: &[usize],
    what: &str,
) -> f32 {
    assert_eq!(matrix.shape(), shape, "{what}: shape differs");
    close_vec(&matrix.iter().copied().collect::<Vec<_>>(), expected, what)
}

fn close_array1(array: &Array1<f64>, expected: &[f32], what: &str) -> f32 {
    let got: Vec<f32> = array.iter().map(|value| *value as f32).collect();
    close_vec(&got, expected, what)
}

#[test]
fn filterbanks_and_frequency_tables_match_the_reference() {
    let dir = golden_dir().join("filters");
    let index = dir.join("index.tsv");
    if !index.is_file() {
        assert!(
            std::env::var("MADMOM_REQUIRE_GOLDEN").is_err(),
            "{} is missing: run tools/dump_filters.py first (see tests/golden/README.md)",
            index.display()
        );
        return;
    }

    let bins = bin_frequencies();
    let bins_view: ArrayView1<'_, f64> = bins.view();
    let mut checked = 0usize;
    let mut worst = 0.0f32;

    for line in std::fs::read_to_string(&index)
        .expect("index must be readable")
        .lines()
        .filter(|line| !line.trim().is_empty())
    {
        let columns: Vec<&str> = line.split('\t').collect();
        let tag = columns[0];
        let (expected, shape) = reference(&dir, tag);
        let diff = match tag {
            "filterbank_log12_unique" => close_filterbank(
                logarithmic_filterbank(bins_view, 12, 30.0, 17000.0, A4, true, true)
                    .unwrap()
                    .matrix(),
                &expected,
                &shape,
                tag,
            ),
            "filterbank_log12_all" => close_filterbank(
                logarithmic_filterbank(bins_view, 12, 30.0, 17000.0, A4, true, false)
                    .unwrap()
                    .matrix(),
                &expected,
                &shape,
                tag,
            ),
            "filterbank_log3_unique" => close_filterbank(
                logarithmic_filterbank(bins_view, 3, 30.0, 17000.0, A4, true, true)
                    .unwrap()
                    .matrix(),
                &expected,
                &shape,
                tag,
            ),
            "filterbank_log6_unique" => close_filterbank(
                logarithmic_filterbank(bins_view, 6, 30.0, 17000.0, A4, true, true)
                    .unwrap()
                    .matrix(),
                &expected,
                &shape,
                tag,
            ),
            "filterbank_mel40_unit" => close_filterbank(
                mel_filterbank(bins_view, 40, 20.0, 17000.0, true, true)
                    .unwrap()
                    .matrix(),
                &expected,
                &shape,
                tag,
            ),
            "filterbank_mel40_all" => close_filterbank(
                mel_filterbank(bins_view, 40, 20.0, 17000.0, true, false)
                    .unwrap()
                    .matrix(),
                &expected,
                &shape,
                tag,
            ),
            "filterbank_bark_normal" => close_filterbank(
                bark_filterbank(bins_view, BarkSpacing::Normal, 20.0, 15500.0, true, true)
                    .unwrap()
                    .matrix(),
                &expected,
                &shape,
                tag,
            ),
            "filterbank_bark_double" => close_filterbank(
                bark_filterbank(bins_view, BarkSpacing::Double, 20.0, 15500.0, true, true)
                    .unwrap()
                    .matrix(),
                &expected,
                &shape,
                tag,
            ),
            "conversions_mel_frequencies_3" => {
                close_array1(&mel_frequencies(3, 20.0, 17000.0), &expected, tag)
            }
            "conversions_mel_frequencies_40" => {
                close_array1(&mel_frequencies(40, 20.0, 17000.0), &expected, tag)
            }
            "conversions_log_frequencies_12" => close_array1(
                &madmom_audio::filters::log_frequencies(12, 30.0, 17000.0, A4).unwrap(),
                &expected,
                tag,
            ),
            "conversions_log_frequencies_3" => close_array1(
                &madmom_audio::filters::log_frequencies(3, 30.0, 17000.0, A4).unwrap(),
                &expected,
                tag,
            ),
            "conversions_semitone_400_500" => close_array1(
                &semitone_frequencies(400.0, 500.0, A4).unwrap(),
                &expected,
                tag,
            ),
            "conversions_bark_normal" => {
                close_array1(&bark_frequencies(20.0, 15500.0), &expected, tag)
            }
            "conversions_bark_double" => {
                close_array1(&bark_double_frequencies(20.0, 15500.0), &expected, tag)
            }
            "conversions_mel_grid" => close_vec(
                &[30.0f64, 100.0, 440.0, 1000.0, 17000.0]
                    .iter()
                    .map(|f| hz2mel(*f) as f32)
                    .collect::<Vec<_>>(),
                &expected,
                tag,
            ),
            "conversions_erb_grid" => close_vec(
                &[30.0f64, 100.0, 440.0, 1000.0, 17000.0]
                    .iter()
                    .map(|f| hz2erb(*f) as f32)
                    .collect::<Vec<_>>(),
                &expected,
                tag,
            ),
            "conversions_midi_grid" => close_vec(
                &[30.0f64, 100.0, 440.0, 1000.0, 17000.0]
                    .iter()
                    .map(|f| hz2midi(*f, A4) as f32)
                    .collect::<Vec<_>>(),
                &expected,
                tag,
            ),
            "conversions_midi_to_hz" => close_vec(
                &[21.0f64, 40.0, 69.0, 100.0, 127.0]
                    .iter()
                    .map(|m| midi2hz(*m, A4) as f32)
                    .collect::<Vec<_>>(),
                &expected,
                tag,
            ),
            "conversions_erb_to_hz" => close_vec(
                &[1.0f64, 5.0, 10.0, 20.0]
                    .iter()
                    .map(|e| erb2hz(*e) as f32)
                    .collect::<Vec<_>>(),
                &expected,
                tag,
            ),
            other => panic!(
                "{other} has a dump but no case in this test: update filters_golden.rs \
                 together with tools/dump_filters.py"
            ),
        };
        worst = worst.max(diff);
        checked += 1;
    }

    assert_eq!(checked, 20, "expected the twenty dumped cases");
    eprintln!("largest |filterbank - reference| = {worst:.3e} (bound {FILTER_ATOL:.0e})");
}

/// Sanity check that the logs cover what the tests compare: a filterbank is a matrix.
#[test]
fn filterbank_applies_to_a_spectrum() {
    let bins = bin_frequencies();
    let filterbank: Filterbank =
        logarithmic_filterbank(bins.view(), 12, 30.0, 17000.0, A4, true, true).unwrap();
    let spectrum = Array1::<f32>::ones(filterbank.num_bins());
    let bands = filterbank.apply(spectrum.view());
    assert_eq!(bands.len(), filterbank.num_bands());
    // every band sums its column, and every column sums to 1 -> 81 bands of 1.0
    for (index, value) in bands.iter().enumerate() {
        assert!((value - 1.0).abs() < 1e-3, "band {index} is {value}");
    }
}
