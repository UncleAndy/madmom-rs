//! WAV reading against the reference dumps (`tests/golden/io/`, see its README).
//!
//! The dumps were produced by madmom's own `load_wave_file` (i.e. scipy's `wavfile`),
//! in the file's native dtype. Every case is compared **bit for bit**: the sample
//! format, the shape, the sample rate and every single sample. A tolerance would hide
//! the conventions this reader exists for — unsigned 8-bit, 24-bit sign extension
//! without rescaling, stereo as `(num_samples, num_channels)`, `start`/`stop` rounded
//! to the nearest sample.

use std::path::{Path, PathBuf};

use madmom_io::audio::{load_wave_file, LoadOptions};
use madmom_io::npy;

/// Directory with the reference dumps (`MADMOM_GOLDEN` overrides it).
fn golden_dir() -> PathBuf {
    std::env::var("MADMOM_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace_root().join("tests").join("golden"))
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate lives in <root>/crates/madmom-io")
        .to_path_buf()
}

/// Directory with madmom's test fixtures (`MADMOM_FIXTURES` overrides it).
fn fixtures_dir() -> PathBuf {
    std::env::var("MADMOM_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace_root().join("third_party/madmom/tests/data"))
}

fn optional_number(text: &str) -> Option<f64> {
    if text.is_empty() {
        None
    } else {
        Some(text.parse().expect("index holds numbers"))
    }
}

fn shape_string(shape: &[usize]) -> String {
    shape
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

/// `fixture + optional .start/.stop markers + .npy`, the naming of tools/dump_audio.py.
fn dump_name(fixture: &str, start: &str, stop: &str) -> String {
    let mut name = fixture.to_string();
    if !start.is_empty() {
        name.push_str(&format!(".start{start}"));
    }
    if !stop.is_empty() {
        name.push_str(&format!(".stop{stop}"));
    }
    name.push_str(".npy");
    name
}

#[test]
fn wave_reading_matches_the_reference_bit_for_bit() {
    let golden = golden_dir().join("io");
    let index = golden.join("index.tsv");
    if !index.is_file() {
        assert!(
            std::env::var("MADMOM_REQUIRE_GOLDEN").is_err(),
            "{} is missing: run tools/dump_audio.py first (see tests/golden/README.md)",
            index.display()
        );
        return;
    }

    let text = std::fs::read_to_string(&index).expect("index must be readable");
    let mut checked = 0usize;
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let columns: Vec<&str> = line.split('\t').collect();
        assert_eq!(columns.len(), 7, "malformed index row: {line}");
        let (fixture, start, stop) = (columns[0], columns[1], columns[2]);
        let expected_rate: u32 = columns[3].parse().expect("sample rate");
        let expected_descr = columns[4];
        let expected_shape = columns[5];

        let options = LoadOptions {
            start: optional_number(start),
            stop: optional_number(stop),
            ..Default::default()
        };
        let audio = load_wave_file(fixtures_dir().join("audio").join(fixture), &options)
            .unwrap_or_else(|e| panic!("{fixture} (start {start:?}, stop {stop:?}): {e}"));

        assert_eq!(audio.sample_rate, expected_rate, "{fixture}: sample rate");
        assert_eq!(
            audio.format.descr(),
            expected_descr,
            "{fixture}: sample format"
        );
        assert_eq!(
            shape_string(audio.samples.shape()),
            expected_shape,
            "{fixture}: shape"
        );

        let reference = npy::read_npy(golden.join(dump_name(fixture, start, stop)))
            .unwrap_or_else(|e| panic!("{fixture}: {e}"));
        let expected = reference.as_f32().expect("reference holds numeric samples");
        let ours: Vec<f32> = audio.samples.iter().copied().collect();
        assert_eq!(ours.len(), expected.len(), "{fixture}: number of samples");
        for (index, (got, want)) in ours.iter().zip(&expected).enumerate() {
            assert_eq!(
                got, want,
                "{fixture}: sample {index} differs ({got} vs {want})"
            );
        }
        checked += 1;
    }
    assert!(
        checked >= 9,
        "expected at least the nine dumped cases, got {checked}"
    );
}
