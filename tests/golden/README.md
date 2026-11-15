# Golden stage dumps

Reference numbers for the port, dumped from the madmom checkout in
`third_party/madmom` (with its Cython extensions built in place). The Rust DSP
stages are compared against these files; `.npy` files are **not committed** —
they are derived data (regenerable in seconds, and madmom's models carry
CC BY-NC-SA 4.0), and a stale copy is worse than none.

## Regenerate

```bash
nix develop                     # or any python with numpy/scipy/cython/mido/setuptools

# 1. build the reference extensions in place (+ egg-info, madmom reads its own
#    version through importlib.metadata)
nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy ps.cython ps.setuptools ps.mido ps.pytest])' \
  --run "python3 tools/build_madmom_ext.py third_party/madmom"

# 2. dump the stages of the flagship chain for one file
nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy ps.mido])' \
  --run "python3 tools/dump_reference.py --audio third_party/madmom/tests/data/audio/sample.wav --out tests/golden"

# 3. optional: the full test-suite baseline of the reference implementation
./tools/build_reference.sh      # writes tools/reference-baseline.txt
```

## Content (`sample.wav`, 44.1 kHz mono, 123 481 samples ≈ 2.8 s)

DSP stages of a **single-resolution chain** (`--frame-size 2048 --num-bands 6`), i.e. one
band bank — that is what the individual Rust processors are pinned against:

| file | dtype | shape | stage |
| --- | --- | --- | --- |
| `sample.signal.npy` | int16 | (123481,) | `SignalProcessor(num_channels=1)` |
| `sample.frames.npy` | int16 | (281, 2048) | `FramedSignalProcessor(frame_size=2048)`, hop 441 |
| `sample.stft.npy` | complex64 | (281, 1024) | `ShortTimeFourierTransformProcessor()` |
| `sample.spec.npy` | float32 | (281, 1024) | magnitude spectrogram |
| `sample.filt.npy` | float32 | (281, 45) | filtered spectrogram, 45 bands |
| `sample.logspec.npy` | float32 | (281, 45) | log filtered spectrogram |
| `sample.diff.npy` | float32 | (281, 90) | positive/negative first differences (45 + 45) |

The last four entries come from the **public processor**, not from that single-resolution
chain, because the networks are fed a wider vector than one band bank produces:

| file | dtype | shape | stage |
| --- | --- | --- | --- |
| `sample.features.npy` | float32 | (281, 266) | what the `BEATS_BLSTM` networks receive (three resolutions) |
| `sample.activations.npy` | float32 | (281,) | ensemble output of `BEATS_BLSTM` |
| `sample.path.npy` | int | (281,) | `DBNBeatTrackingProcessor.hmm.viterbi` state path |
| `sample.beats.npy` | float32 | (8,) | beat positions in seconds |

`dump_preproc.py --group beats_lstm` shows the other width: the online chain feeds its
networks **162** features per frame (12 bands, `log10`, `SpectrogramDifferenceProcessor`).

## Cross-checked against madmom's own fixtures

`sample.activations.npy` versus `third_party/madmom/tests/data/activations/sample.beats_blstm.npz`:
max |Δ| = 8.9e-08, mean 1.1e-08 — agreement at float32 rounding level (the fixture was produced
by a different madmom run, so bit-equality is not expected; the plan's tolerance for network
activations is 1e-4). `sample.features.npy` is additionally identical to the input dumped
independently by `dump_preproc.py --preproc` (two code paths, same numbers).

## Audio decoding dumps (`io/`)

Produced by `tools/dump_audio.py` from madmom's own `load_wave_file` (scipy's
`wavfile`), in the file's native dtype:

```bash
nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy ps.mido])' \
  --run "python3 tools/dump_audio.py third_party/madmom"
```

| case | dtype | shape | rate |
| --- | --- | --- | --- |
| `sample.wav` | int16 | (123481,) | 44100 |
| `sample_22050.wav` | int16 | (61741,) | 22050 |
| `sample2.wav` | int16 | (178636,) | 44100 |
| `stereo_sample.wav` | int16 | (182919, 2) | 44100 |
| `stereo_chirp.wav` | int16 | (22050, 2) | 44100 |
| `sample.wav` + `start=0.1`, `start=0.1 stop=0.5`, `stop=0.25` | int16 | (119071,) / (17640,) / (11025,) | 44100 |
| `stereo_sample.wav` + `start=0.2 stop=0.6` | int16 | (17640, 2) | 44100 |

`index.tsv` lists `fixture, start, stop, sample_rate, descr, shape, sha256`;
`crates/madmom-io/tests/audio_golden.rs` replays every row through the Rust WAV reader
and requires the format, shape, rate and **all samples** to be identical (verified:
bit-exact on all nine cases).

## Spectrogram chain (`sample.*.npy`)

`crates/madmom-audio/tests/spectrogram_golden.rs` replays the dumped chain stage by
stage (magnitudes → logarithmic filterbank, 6 bands per octave → `log10(1 + x)` →
positive differences, ratio 0.5, stacked horizontally) and compares every stage against
its own dump. Observed differences with a 1e-5 bound:

| stage | max abs diff |
| --- | --- |
| `spec` (`\|STFT\|`) | 7.629e-06 |
| `filt` (filterbank) | 7.629e-06 |
| `logspec` (`log10(1+x)`) | 1.192e-07 |
| `diff` (positive, hstacked) | 2.384e-07 |

The 7.6e-06 is exactly **one float32 ulp** at the magnitudes involved (64–128): the two
FFT implementations agree in float64, and casting their results into `complex64` flips
the last bit for a small fraction of the values. That is why the plan's tolerance for
spectrograms is 1e-5 and not 1e-9.

## Filterbanks and frequency tables (`filters/`)

Produced by `tools/dump_filters.py` from madmom's own filterbanks, with the bin
frequencies of the flagship chains (2048-point FFT at 44.1 kHz, 1024 bins):

```bash
nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy ps.mido])' \
  --run "python3 tools/dump_filters.py third_party/madmom"
```

Twenty cases: `LogarithmicFilterbank` (12/6/3 bands per octave, with and without bin
deduplication), `MelFilterbank` (40 bands, both settings), `BarkFilterbank` (normal and
double spacing), plus the frequency tables and scalings (`mel_frequencies`,
`log_frequencies`, `semitone_frequencies`, `bark_frequencies`,
`bark_double_frequencies`, `hz2mel`, `hz2erb`, `hz2midi`, `midi2hz`, `erb2hz`).

`crates/madmom-audio/tests/filters_golden.rs` builds every case from the tag names and
requires agreement; the largest difference observed is **1.490e-08** (bound 1e-6), which
is the float32 normalization sum being added up in a different order than NumPy's
pairwise sum. The dumps already caught one transcription error in this port (the double
Bark table had 60 Hz where madmom has 50 Hz).

## Joint beat/downbeat DBN (`downbeats/`)

Produced by `tools/dump_downbeats.py`. The input is **not** audio but the reference's own
network output, `tests/data/activations/sample.downbeats_blstm.npz` (281 frames at
100 fps, two columns: beat and downbeat), so the DBN is pinned without needing Triton:

```bash
nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy ps.mido])' \
  --run "python3 tools/dump_downbeats.py third_party/madmom"
```

Cases: `result.npy` (the 8 (down-)beats with their beat number, what the processor
returns), `trimmed.npy` (the activations after the 0.05 threshold — rows 7..267 of 281),
`path_{3,4}.npy`/`logprob_{3,4}.npy` (Viterbi path and log probability of each modelled
bar length on the untrimmed activations) and
`path_trimmed_{3,4}.npy`/`logprob_trimmed_{3,4}.npy` (the same on the trimmed ones, which
is what the processor actually decodes).

`crates/madmom-features/tests/downbeats_golden.rs` requires the same state paths (281 of
281 and 261 of 261 frames, exact integers), the same log probabilities
(-764.586594 for 3/4 and -760.212077 for 4/4) and the same beats —
`[(0.09, 1), (0.45, 2), (0.79, 3), (1.12, 4), (1.47, 1), (1.8, 2), (2.14, 3), (2.49, 4)]`.

Two traps this dump already paid for:

* `np.argmax` over the two-column activation window is a **flat** argmax — a later row wins
  on its downbeat column even when its beat column is smaller. Comparing the rows
  lexicographically gave 0.08 s instead of 0.09 s for the first beat.
* madmom returns `np.vstack(...).T`, so `np.save` writes the array **Fortran-ordered**
  (`fortran_order: True`). A reader that ignores the flag hands out a transposed array;
  `madmom-io` now reorders on read (`reads_fortran_ordered_arrays_in_c_order`).

## Tempo estimation (`tempo/`)

Produced by `tools/dump_tempo.py`, which walks madmom's own processors over the reference
activation fixtures:

```bash
nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy ps.mido])' \
  --run "python3 tools/dump_tempo.py third_party/madmom"
```

Twenty-six cases on `sample.beats_blstm.npz` (281 frames at 100 fps) plus
`sample.beats_tcn_tempo.npz` for the multi-task TCN tempo layer: the comb and
autocorrelation interval histograms (function level and through
`CombFilterTempoHistogramProcessor` / `ACFTempoHistogramProcessor`), the DBN one, the
smoothed variants, the smoothed activations, `dominant_interval`, the detected tempi of all
three frame-based sources, and the TCN histogram and tempi (default and `hist_smooth=15`).

`crates/madmom-features/tests/tempo_golden.rs` requires agreement within the plan's `f64`
bounds; the largest difference observed is **8.882e-16** (the comb histogram), the DBN
histogram is exact. The dumps already paid for one trap: the reference's Cython helper
declares `float alpha`, so the backward comb filter runs with a **float32-rounded** scaling
factor while the signal stays float64. Ignoring that rounding put the comb histogram
4.378e-07 away from the reference (and the smoothed histogram just outside the 1e-6 bound).

`interpolate=True` is refused with a typed error, not approximated: it needs scipy's
quadratic B-spline interpolation (`interp1d(..., 'quadratic')`). Its dumps
(`detect_tempo_hist_interp.npy`, `tempi_comb_interp.npy`) are in place, so the gap is
measurable when that lands.

## Onset detection and peak picking (`onsets/`)

Produced by `tools/dump_onsets.py`, in two independent groups:

```bash
nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy ps.mido])' \
  --run "python3 tools/dump_onsets.py third_party/madmom"
```

* the eleven onset detection functions, evaluated on the chain the reference's own onset
  tests build — `Spectrogram(sample.wav, circular_shift=True)` and its logarithmic filtered
  version (24 bands per octave, which is 140 bands over the audible range, `mul=1`,
  `add=1`) — together with that spectrogram, the filtered one and the phase, so the input is
  checked before the ODFs are. The phase-weighted ODFs (`weighted_phase_deviation`,
  `normalized_weighted_phase_deviation`, `complex_domain`, `rectified_complex_domain`) are
  dumped from the **unfiltered** spectrogram, because the reference raises on a filtered
  one (its phase has the unfiltered bin count);
* `peak_picking` and `OnsetPeakPickingProcessor` on madmom's activation fixtures
  (`sample.super_flux.npz` at 200 fps, `sample.onsets_rnn.npz` at 100 fps), including the
  online mode both all-at-once and frame by frame.

`crates/madmom-features/tests/onsets_golden.rs` requires agreement within the plan's bounds.
The largest differences are 4.6e-07 (`hfc`), 2.0e-06 (`spectral_flux`), 4.3e-06
(`complex_flux`) and 1.2e-04 for the complex-domain ODFs — the last one because madmom
computes `exp(1j * phase)` in complex64 while the port does it in float64, on values around
500. The peak frames and every onset (offline, online all-at-once and online frame by frame)
match **exactly**.

The dumps caught two shape assumptions: `phase_deviation` and `complex_flux` must *not*
require the phase to have the spectrogram's shape (the reference reaches the phase through
the unfiltered transform), while the phase-weighted ODFs must. They also pinned a detail of
`_complex_domain`: the first frame's target is the bare magnitude, not the magnitude with
its own phase.

## Verified framing rule

`FramedSignal` (defaults `END_OF_SIGNAL='normal'`, `origin='center'`):

```
num_frames = ceil(num_samples / hop_size)        # hop = sample_rate / fps = 441
frame_i    = signal[i*hop - frame_size//2 : i*hop + frame_size//2]   # centered!
samples outside the signal are zeros (signal_frame(pad=0))
```

Reimplemented independently in numpy from that rule, `sample.frames.npy` is
reproduced **bit-exactly** (281 frames, int16). Two traps worth remembering:

* frames are **centered** on `i*hop` — left-aligned framing silently shifts
  every stage by 1024 samples;
* the number of frames is `ceil(num_samples/hop)`, i.e. the trailing frames are
  padded, not dropped (`1 + (n - frame_size)//hop` = 276 here, which is wrong).
