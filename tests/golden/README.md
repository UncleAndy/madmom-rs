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
