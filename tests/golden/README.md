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

| file | dtype | shape | stage |
| --- | --- | --- | --- |
| `sample.signal.npy` | int16 | (123481,) | `SignalProcessor(num_channels=1)` |
| `sample.frames.npy` | int16 | (281, 2048) | `FramedSignalProcessor(frame_size=2048)`, hop 441 |
| `sample.stft.npy` | complex64 | (281, 1024) | `ShortTimeFourierTransformProcessor()` |
| `sample.spec.npy` | float32 | (281, 1024) | magnitude spectrogram |
| `sample.filt.npy` | float32 | (281, 45) | filtered spectrogram, 45 bands |
| `sample.logspec.npy` | float32 | (281, 45) | log filtered spectrogram |
| `sample.diff.npy` | float32 | (281, 90) | positive/negative first differences (45 + 45) |
| `sample.features.npy` | float32 | (25290,) | flattened NN input: 281 frames x 90 features |

The activation/path/beat stages are not dumped here: they belong to the mm
pipeline and are covered by madmom's own `.npz` fixtures
(`third_party/madmom/tests/data/activations/*.npz`), which the Rust tests read
directly — see plan §3.3.

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
