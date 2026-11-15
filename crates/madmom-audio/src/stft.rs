//! Short-time Fourier transform.
//!
//! Port of `madmom/audio/stft.py` (711 lines upstream, 27 tests in
//! `tests/test_audio_stft.py`). The numeric traps, all of them silent:
//!
//! * the window is scaled by `1 / np.iinfo(dtype).max` (32767 for int16) when the
//!   frames come from an integer signal — forgot it and every magnitude is off by a
//!   factor of 32767 while detectors still return plausible results;
//! * madmom computes in **float64** (int16 frames times a float64 window) and only
//!   casts the result into the `complex64` array, so the FFT has to run in `f64` here
//!   as well, even though the stored values are `complex32`;
//! * `num_bins` is `fft_size / 2`, i.e. the Nyquist bin is dropped unless
//!   `include_nyquist` is set (2048-point frames give 1024 bins, not 1025);
//! * `circular_shift` swaps the frame halves *before* the FFT (needed for phase), and
//!   pads between the halves when `fft_size > frame_size`, while a shorter `fft_size`
//!   **truncates** the frame (`fftpack.fft` semantics) instead of refusing it.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::{Complex, Complex32};
use rustfft::FftPlanner;
use thiserror::Error;

use crate::signal::{DType, FramedSignal};

/// Data type of the stored transform (`STFT_DTYPE` upstream).
pub type StftValue = Complex32;

/// Errors of the STFT stage.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum StftError {
    /// Frames must be a 2-D array `(num_frames, frame_size)`.
    #[error("frames must be a 2D array, got {0} dimensions")]
    FramesNotTwoDimensional(usize),
    /// An FFT size smaller than the shift would index out of bounds.
    #[error("fft_size {fft_size} is too small for circular shifting a frame of {frame_size}")]
    FftSizeTooSmall {
        /// Requested FFT size.
        fft_size: usize,
        /// Frame size in samples.
        frame_size: usize,
    },
    /// An explicit window must have one coefficient per sample of a frame.
    #[error("window has {window} coefficients, frames have {frame_size} samples")]
    WindowLengthMismatch {
        /// Number of coefficients in the window.
        window: usize,
        /// Frame size in samples.
        frame_size: usize,
    },
}

/// Window function that is materialised for the frame size (madmom passes callables).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunction {
    /// `np.hanning`, the default window of the reference.
    Hanning,
    /// `np.hamming`.
    Hamming,
    /// A rectangular window, i.e. `np.ones`.
    Rectangular,
}

impl WindowFunction {
    /// Materialise the window for `size` samples (in `f64`, like NumPy).
    pub fn build(self, size: usize) -> Array1<f64> {
        match self {
            WindowFunction::Hanning => hanning(size),
            WindowFunction::Hamming => hamming(size),
            WindowFunction::Rectangular => Array1::ones(size),
        }
    }
}

/// Window argument of the transform: a function, an explicit window, or none.
#[derive(Debug, Clone)]
pub enum Window {
    /// A window function, materialised for the frame size.
    Function(WindowFunction),
    /// An explicit window of length `frame_size`.
    Array(Array1<f64>),
    /// No window at all (the reference's `window=None`).
    None,
}

impl Window {
    /// Materialise the window for `size` samples, if one is set.
    fn materialise(&self, size: usize) -> Option<Array1<f64>> {
        match self {
            Window::Function(function) => Some(function.build(size)),
            Window::Array(array) => Some(array.clone()),
            Window::None => None,
        }
    }
}

/// A symmetric Hann window, exactly like `np.hanning`.
///
/// `0.5 - 0.5 * cos(2*pi*n/(size-1))`, with NumPy's special cases for `size <= 1`.
pub fn hanning(size: usize) -> Array1<f64> {
    let mut window = Array1::zeros(size);
    if size == 0 {
        return window;
    }
    if size == 1 {
        window[0] = 1.0;
        return window;
    }
    let denominator = (size - 1) as f64;
    for (n, value) in window.iter_mut().enumerate() {
        *value = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / denominator).cos();
    }
    window
}

/// A symmetric Hamming window, exactly like `np.hamming`.
pub fn hamming(size: usize) -> Array1<f64> {
    let mut window = Array1::zeros(size);
    if size == 0 {
        return window;
    }
    if size == 1 {
        window[0] = 1.0;
        return window;
    }
    let denominator = (size - 1) as f64;
    for (n, value) in window.iter_mut().enumerate() {
        *value = 0.54 - 0.46 * (2.0 * std::f64::consts::PI * n as f64 / denominator).cos();
    }
    window
}

/// Frequencies of the FFT bins in Hz, following `fft_frequencies`:
/// `np.fft.fftfreq(num_fft_bins * 2, 1 / sample_rate)[:num_fft_bins]`, i.e. the
/// non-negative half of the frequency grid.
pub fn fft_frequencies(num_fft_bins: usize, sample_rate: f64) -> Array1<f64> {
    let total = num_fft_bins * 2;
    let step = 1.0 / (total as f64 * (1.0 / sample_rate));
    (0..num_fft_bins).map(|bin| bin as f64 * step).collect()
}

/// Complex short-time Fourier transform of a framed signal.
///
/// `window` is applied *as given*; the `1 / iinfo(dtype).max` scaling happens in
/// [`ShortTimeFourierTransform::new`], exactly as in the reference, so that this
/// function can also be fed an already scaled window.
pub fn stft(
    frames: ArrayView2<'_, f64>,
    window: Option<ArrayView1<'_, f64>>,
    fft_size: Option<usize>,
    circular_shift: bool,
    include_nyquist: bool,
) -> Result<Array2<StftValue>, StftError> {
    if frames.ndim() != 2 {
        return Err(StftError::FramesNotTwoDimensional(frames.ndim()));
    }
    let (num_frames, frame_size) = (frames.shape()[0], frames.shape()[1]);
    let fft_size = fft_size.unwrap_or(frame_size);
    let mut num_bins = fft_size >> 1;
    if include_nyquist {
        num_bins += 1;
    }
    let shift = frame_size >> 1;
    if circular_shift && fft_size < shift {
        return Err(StftError::FftSizeTooSmall {
            fft_size,
            frame_size,
        });
    }
    if let Some(window) = window {
        if window.len() != frame_size {
            return Err(StftError::WindowLengthMismatch {
                window: window.len(),
                frame_size,
            });
        }
    }

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let mut buffer = vec![Complex::new(0.0f64, 0.0); fft_size];
    let mut output = Array2::<StftValue>::zeros((num_frames, num_bins));

    for frame_index in 0..num_frames {
        let frame = frames.row(frame_index);
        // windowed samples of this frame (float64: madmom multiplies its int16 frames
        // with a float64 window and only casts the transform afterwards)
        let mut signal = vec![0.0f64; frame_size];
        match window {
            Some(window) => {
                for (index, value) in signal.iter_mut().enumerate() {
                    *value = frame[index] * window[index];
                }
            }
            None => signal.copy_from_slice(frame.as_slice().expect("row is contiguous")),
        }
        buffer
            .iter_mut()
            .for_each(|value| *value = Complex::new(0.0, 0.0));
        if circular_shift {
            // swap the two halves of the windowed signal, zero padding in between when
            // the FFT is longer than the frame
            for index in 0..shift {
                buffer[index] = Complex::new(signal[shift + index], 0.0);
                buffer[fft_size - shift + index] = Complex::new(signal[index], 0.0);
            }
        } else {
            // `fftpack.fft(signal, fft_size)` pads with zeros when the FFT is longer
            // than the frame and *truncates* the frame when it is shorter.
            for (index, value) in signal.iter().take(fft_size).enumerate() {
                buffer[index] = Complex::new(*value, 0.0);
            }
        }
        fft.process(&mut buffer);
        for bin in 0..num_bins {
            output[[frame_index, bin]] =
                Complex32::new(buffer[bin].re as f32, buffer[bin].im as f32);
        }
    }
    Ok(output)
}

/// Phase of a transform, `np.angle`: the argument in `[-pi, pi]`.
#[allow(clippy::excessive_precision)]
pub fn phase(transform: ArrayView2<'_, StftValue>) -> Array2<f64> {
    transform.mapv(|value| (value.im as f64).atan2(value.re as f64))
}

/// Short-time Fourier transform of a framed signal, with the window scaled the way
/// the reference scales it.
#[derive(Debug, Clone)]
pub struct ShortTimeFourierTransform {
    values: Array2<StftValue>,
    frames: FramedSignal,
    window: Window,
    fft_window: Option<Array1<f64>>,
    fft_size: usize,
    circular_shift: bool,
    include_nyquist: bool,
}

impl ShortTimeFourierTransform {
    /// Compute the transform of `frames`.
    ///
    /// The window is scaled by `1 / iinfo(dtype).max` when the frames come from an
    /// integer signal; a float signal uses the window unscaled, and `Window::None`
    /// becomes a scaled rectangular window for integer signals (or no window at all
    /// for float signals), exactly like the reference's `try/except` cascade.
    pub fn new(
        frames: FramedSignal,
        window: Window,
        fft_size: Option<usize>,
        circular_shift: bool,
        include_nyquist: bool,
    ) -> Result<Self, StftError> {
        let frame_size = frames.frame_size();
        let materialised = window.materialise(frame_size);
        let dtype = frames.signal().dtype();
        let fft_window = if dtype.is_integer() {
            let max_range = dtype.max_amplitude();
            Some(match &materialised {
                Some(window) => window.mapv(|value| value / max_range),
                None => Array1::<f64>::ones(frame_size).mapv(|value| value / max_range),
            })
        } else {
            materialised
        };
        let as_f64 = frames.to_array().mapv(|value| value as f64);
        let values = stft(
            as_f64.view(),
            fft_window.as_ref().map(|window| window.view()),
            fft_size,
            circular_shift,
            include_nyquist,
        )?;
        Ok(Self {
            values,
            fft_size: fft_size.unwrap_or(frame_size),
            frames,
            window,
            fft_window,
            circular_shift,
            include_nyquist,
        })
    }

    /// The transform as `(num_frames, num_bins)` complex values.
    pub fn values(&self) -> ndarray::ArrayView2<'_, StftValue> {
        self.values.view()
    }

    /// Number of frames.
    pub fn num_frames(&self) -> usize {
        self.values.shape()[0]
    }

    /// Number of frequency bins.
    pub fn num_bins(&self) -> usize {
        self.values.shape()[1]
    }

    /// FFT size that was used.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// The framed signal this transform was computed from.
    pub fn frames(&self) -> &FramedSignal {
        &self.frames
    }

    /// Window that was used (before the dtype scaling).
    pub fn window(&self) -> &Window {
        &self.window
    }

    /// Window actually multiplied with the frames (after the dtype scaling).
    pub fn fft_window(&self) -> Option<&Array1<f64>> {
        self.fft_window.as_ref()
    }

    /// Whether the frames were circularly shifted before the FFT.
    pub fn circular_shift(&self) -> bool {
        self.circular_shift
    }

    /// Whether the Nyquist bin is included.
    pub fn include_nyquist(&self) -> bool {
        self.include_nyquist
    }

    /// Frequencies of the bins in Hz.
    pub fn bin_frequencies(&self) -> Array1<f64> {
        let sample_rate = self.frames.signal().sample_rate().unwrap_or(44_100) as f64;
        fft_frequencies(self.num_bins(), sample_rate)
    }

    /// Phases of the transform.
    pub fn phase(&self) -> Array2<f64> {
        phase(self.values())
    }
}

/// Processor that turns a framed signal into a transform, mirroring
/// `ShortTimeFourierTransformProcessor`.
#[derive(Debug, Clone)]
pub struct ShortTimeFourierTransformProcessor {
    /// Window (function, array or none).
    pub window: Window,
    /// FFT size, `None` to use the frame size.
    pub fft_size: Option<usize>,
    /// Circularly shift frames before the FFT (needed for phase).
    pub circular_shift: bool,
    /// Include the Nyquist bin.
    pub include_nyquist: bool,
}

impl Default for ShortTimeFourierTransformProcessor {
    fn default() -> Self {
        Self {
            window: Window::Function(WindowFunction::Hanning),
            fft_size: None,
            circular_shift: false,
            include_nyquist: false,
        }
    }
}

impl ShortTimeFourierTransformProcessor {
    /// Compute the transform of `frames`.
    pub fn process(&self, frames: FramedSignal) -> Result<ShortTimeFourierTransform, StftError> {
        ShortTimeFourierTransform::new(
            frames,
            self.window.clone(),
            self.fft_size,
            self.circular_shift,
            self.include_nyquist,
        )
    }
}

/// Whether a dtype needs the `1 / max` window scaling (integer formats do).
pub fn window_scaling(dtype: DType) -> Option<f64> {
    if dtype.is_integer() {
        Some(1.0 / dtype.max_amplitude())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn hanning_has_the_numpy_shape() {
        // np.hanning(3) == array([0., 1., 0.]) and np.hanning(5) is symmetric
        let window = hanning(3);
        assert!((window[0] - 0.0).abs() < 1e-12);
        assert!((window[1] - 1.0).abs() < 1e-12);
        let window = hanning(5);
        assert!((window[0] - window[4]).abs() < 1e-12);
        assert!((window[1] - window[3]).abs() < 1e-12);
        assert!(window[2] > window[1]);
    }

    #[test]
    fn fft_frequencies_follow_the_bin_spacing() {
        // np.fft.fftfreq(10, 1/10)[:5] == [0., 1., 2., 3., 4.]
        let frequencies = fft_frequencies(5, 10.0);
        assert_eq!(frequencies.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn transform_of_a_constant_frame_puts_all_energy_in_bin_zero() {
        // a rectangular window and a constant signal: only the DC bin is non-zero
        let frames = Array2::from_shape_vec((1, 4), vec![1.0f64; 4]).unwrap();
        let window = WindowFunction::Rectangular.build(4);
        let transform = stft(frames.view(), Some(window.view()), None, false, false).unwrap();
        assert_eq!(transform.shape(), &[1, 2]);
        assert!((transform[[0, 0]].re - 4.0).abs() < 1e-12);
        assert!(transform[[0, 1]].norm() < 1e-12);
    }

    #[test]
    fn nyquist_bin_is_dropped_unless_requested() {
        let frames = Array2::from_shape_vec((1, 8), vec![1.0f64; 8]).unwrap();
        let without = stft(frames.view(), None, None, false, false).unwrap();
        let with = stft(frames.view(), None, None, false, true).unwrap();
        assert_eq!(without.shape(), &[1, 4]);
        assert_eq!(with.shape(), &[1, 5]);
    }

    #[test]
    fn window_scaling_is_the_inverse_of_fft_window() {
        assert_eq!(window_scaling(DType::Int16), Some(1.0 / 32767.0));
        assert_eq!(window_scaling(DType::Float32), None);
    }

    #[test]
    fn window_length_is_checked() {
        let frames = Array2::from_shape_vec((1, 4), vec![0.0f64; 4]).unwrap();
        let window = arr1(&[0.5f64, 0.5]);
        assert_eq!(
            stft(frames.view(), Some(window.view()), None, false, false).unwrap_err(),
            StftError::WindowLengthMismatch {
                window: 2,
                frame_size: 4
            }
        );
    }
}
