//! The `scipy.ndimage` filters madmom uses, with scipy's exact window placement.
//!
//! Three functions cover every use in the port so far: [`maximum_filter`] (SuperFlux's
//! vibrato suppression, the complex-flux group-delay mask, the peak-picking moving
//! maximum), [`minimum_filter`] (the same mask when the spectrogram is not filtered) and
//! [`uniform_filter`] (the moving average of the peak picker).
//!
//! Two conventions decide the numbers, and neither is obvious:
//!
//! * **The window is placed at `i - origin - size/2 ..= i - origin - size/2 + size - 1`.**
//!   A positive `origin` shifts the window *left*, as scipy documents. Verified against
//!   scipy for sizes 1..5 and origins -1..+2 (`maximum_filter(np.arange(1, 9), size=5,
//!   origin=2)` starts `[4, 3, 3, 4, ...]`, which only this placement reproduces).
//! * **`mode='reflect'` is edge-symmetric** (`d c b a | a b c d | d c b a`), i.e. numpy's
//!   *symmetric* padding, not numpy's `reflect` (which would drop the edge sample). The
//!   default of every scipy filter is this mode, so a naive edge clamp is wrong by one
//!   sample everywhere near a border.
//!
//! `mode='constant'` uses `cval` (0.0 by default) outside the array **and still divides by
//! the full window size** in [`uniform_filter`], which is what makes the moving average of
//! the peak picker dip at the start of the activations.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// How indices outside the array are filled, i.e. scipy's `mode`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    /// Edge-symmetric reflection (`scipy`'s `'reflect'`, the default of every filter).
    Reflect,
    /// A constant value outside the array (`mode='constant'`, `cval=0.0` in madmom).
    Constant(f64),
}

impl Mode {
    /// `mode='constant'` with scipy's default `cval=0.0`.
    pub fn constant() -> Self {
        Mode::Constant(0.0)
    }
}

/// Map an out-of-range index into `0..len` for [`Mode::Reflect`], repeating if needed.
fn reflect_index(index: i64, len: usize) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let len = len as i64;
    let mut index = index;
    // edge-symmetric reflection: -1 -> 0, -2 -> 1, len -> len-1, len+1 -> len-2
    while index < 0 || index >= len {
        if index < 0 {
            index = -index - 1;
        } else {
            index = 2 * len - 1 - index;
        }
    }
    Some(index as usize)
}

/// The input indices a filter window of `size` centred on `index` with `origin` covers.
fn window(index: usize, size: usize, origin: i64) -> std::ops::Range<i64> {
    let start = index as i64 - origin - (size / 2) as i64;
    start..start + size as i64
}

fn value_at_1d(values: &[f64], index: i64, mode: Mode) -> f64 {
    match mode {
        Mode::Reflect => reflect_index(index, values.len())
            .map(|index| values[index])
            .unwrap_or(0.0),
        Mode::Constant(cval) => {
            if index < 0 || index as usize >= values.len() {
                cval
            } else {
                values[index as usize]
            }
        }
    }
}

fn value_at_2d(values: ArrayView2<'_, f64>, row: i64, column: i64, mode: Mode) -> f64 {
    let (rows, columns) = (values.shape()[0], values.shape()[1]);
    match mode {
        Mode::Reflect => {
            let row = reflect_index(row, rows);
            let column = reflect_index(column, columns);
            match (row, column) {
                (Some(row), Some(column)) => values[[row, column]],
                _ => 0.0,
            }
        }
        Mode::Constant(cval) => {
            if row < 0 || row as usize >= rows || column < 0 || column as usize >= columns {
                cval
            } else {
                values[[row as usize, column as usize]]
            }
        }
    }
}

/// `scipy.ndimage.maximum_filter` for a 1-D array.
pub fn maximum_filter_1d(
    values: ArrayView1<'_, f64>,
    size: usize,
    origin: i64,
    mode: Mode,
) -> Array1<f64> {
    let mut out = Array1::zeros(values.len());
    for (index, value) in out.iter_mut().enumerate() {
        let mut best = f64::NEG_INFINITY;
        for position in window(index, size, origin) {
            let candidate = value_at_1d(values.as_slice().expect("contiguous"), position, mode);
            if candidate > best {
                best = candidate;
            }
        }
        *value = best;
    }
    out
}

/// `scipy.ndimage.minimum_filter` for a 1-D array.
pub fn minimum_filter_1d(
    values: ArrayView1<'_, f64>,
    size: usize,
    origin: i64,
    mode: Mode,
) -> Array1<f64> {
    let mut out = Array1::zeros(values.len());
    for (index, value) in out.iter_mut().enumerate() {
        let mut best = f64::INFINITY;
        for position in window(index, size, origin) {
            let candidate = value_at_1d(values.as_slice().expect("contiguous"), position, mode);
            if candidate < best {
                best = candidate;
            }
        }
        *value = best;
    }
    out
}

/// `scipy.ndimage.uniform_filter` for a 1-D array (the mean over the window).
pub fn uniform_filter_1d(
    values: ArrayView1<'_, f64>,
    size: usize,
    origin: i64,
    mode: Mode,
) -> Array1<f64> {
    let mut out = Array1::zeros(values.len());
    for (index, value) in out.iter_mut().enumerate() {
        let mut sum = 0.0;
        for position in window(index, size, origin) {
            sum += value_at_1d(values.as_slice().expect("contiguous"), position, mode);
        }
        *value = sum / size as f64;
    }
    out
}

/// `scipy.ndimage.maximum_filter` for a 2-D array, with one size and origin per axis.
pub fn maximum_filter_2d(
    values: ArrayView2<'_, f64>,
    size: (usize, usize),
    origin: (i64, i64),
    mode: Mode,
) -> Array2<f64> {
    let (rows, columns) = (values.shape()[0], values.shape()[1]);
    let mut out = Array2::zeros((rows, columns));
    for row in 0..rows {
        for column in 0..columns {
            let mut best = f64::NEG_INFINITY;
            for candidate_row in window(row, size.0, origin.0) {
                for candidate_column in window(column, size.1, origin.1) {
                    let candidate = value_at_2d(values, candidate_row, candidate_column, mode);
                    if candidate > best {
                        best = candidate;
                    }
                }
            }
            out[[row, column]] = best;
        }
    }
    out
}

/// `scipy.ndimage.minimum_filter` for a 2-D array.
pub fn minimum_filter_2d(
    values: ArrayView2<'_, f64>,
    size: (usize, usize),
    origin: (i64, i64),
    mode: Mode,
) -> Array2<f64> {
    let (rows, columns) = (values.shape()[0], values.shape()[1]);
    let mut out = Array2::zeros((rows, columns));
    for row in 0..rows {
        for column in 0..columns {
            let mut best = f64::INFINITY;
            for candidate_row in window(row, size.0, origin.0) {
                for candidate_column in window(column, size.1, origin.1) {
                    let candidate = value_at_2d(values, candidate_row, candidate_column, mode);
                    if candidate < best {
                        best = candidate;
                    }
                }
            }
            out[[row, column]] = best;
        }
    }
    out
}

/// `scipy.ndimage.uniform_filter` for a 2-D array.
pub fn uniform_filter_2d(
    values: ArrayView2<'_, f64>,
    size: (usize, usize),
    origin: (i64, i64),
    mode: Mode,
) -> Array2<f64> {
    let (rows, columns) = (values.shape()[0], values.shape()[1]);
    let mut out = Array2::zeros((rows, columns));
    let divisor = (size.0 * size.1) as f64;
    for row in 0..rows {
        for column in 0..columns {
            let mut sum = 0.0;
            for candidate_row in window(row, size.0, origin.0) {
                for candidate_column in window(column, size.1, origin.1) {
                    sum += value_at_2d(values, candidate_row, candidate_column, mode);
                }
            }
            out[[row, column]] = sum / divisor;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// The values below are scipy's own outputs for `np.arange(1, 9)`, quoted from a run
    /// of `scipy.ndimage` (see the module documentation for the commands).
    #[test]
    fn maximum_filter_matches_scipy_window_placement() {
        let a = Array1::from((1..=8).map(|value| value as f64).collect::<Vec<_>>());
        let cases: [(usize, i64, [f64; 8]); 6] = [
            (3, 0, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0]),
            (3, 1, [2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            (3, -1, [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 8.0]),
            (4, 0, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0]),
            (4, 1, [3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            (5, 2, [4.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        ];
        for (size, origin, expected) in cases {
            let got = maximum_filter_1d(a.view(), size, origin, Mode::Reflect);
            assert_eq!(
                got.to_vec(),
                expected.to_vec(),
                "size {size}, origin {origin}"
            );
        }
    }

    #[test]
    fn uniform_filter_divides_by_the_full_window() {
        let a = Array1::from((1..=8).map(|value| value as f64).collect::<Vec<_>>());
        // scipy: uniform_filter(a, size=3, mode='constant') -> [1, 2, ..., 7, 5]
        let constant = uniform_filter_1d(a.view(), 3, 0, Mode::constant());
        assert_eq!(constant[0], 1.0);
        assert_eq!(constant[7], 5.0, "the outside samples count as zero");
        // scipy: uniform_filter(a, size=3, origin=1) -> [1.333, 1.333, 2, ...]
        let reflect = uniform_filter_1d(a.view(), 3, 1, Mode::Reflect);
        assert!((reflect[0] - 4.0 / 3.0).abs() < 1e-12);
        assert!((reflect[1] - 4.0 / 3.0).abs() < 1e-12);
        assert_eq!(reflect[2], 2.0);
    }

    #[test]
    fn minimum_filter_matches_scipy() {
        let a = Array1::from((1..=8).map(|value| value as f64).collect::<Vec<_>>());
        let got = minimum_filter_1d(a.view(), 3, 0, Mode::Reflect);
        assert_eq!(got.to_vec(), vec![1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn two_dimensional_filters_use_one_window_per_axis() {
        let m = Array2::from_shape_vec(
            (3, 4),
            (1..=12).map(|value| value as f64).collect::<Vec<_>>(),
        )
        .unwrap();
        let maximum = maximum_filter_2d(m.view(), (1, 3), (0, 0), Mode::Reflect);
        assert_eq!(maximum.row(0).to_vec(), vec![2.0, 3.0, 4.0, 4.0]);
        assert_eq!(maximum.row(1).to_vec(), vec![6.0, 7.0, 8.0, 8.0]);
        let uniform = uniform_filter_2d(m.view(), (3, 1), (1, 0), Mode::constant());
        assert!((uniform[[0, 0]] - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(uniform.row(2).to_vec(), vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn reflect_repeats_for_windows_wider_than_the_array() {
        // a size-3 window on a two-element array reflects more than once
        let a = array![1.0, 2.0];
        let got = maximum_filter_1d(a.view(), 3, 0, Mode::Reflect);
        assert_eq!(got.to_vec(), vec![2.0, 2.0]);
        // scipy's mapping for a 2-element array, measured with map_coordinates:
        // -6 -> 1, -5 -> 0, 3 -> 0, 4 -> 0, 5 -> 1, 6 -> 1
        assert_eq!(reflect_index(-6, 2), Some(1));
        assert_eq!(reflect_index(-5, 2), Some(0));
        assert_eq!(reflect_index(3, 2), Some(0));
        assert_eq!(reflect_index(4, 2), Some(0));
        assert_eq!(reflect_index(5, 2), Some(1));
        assert_eq!(reflect_index(6, 2), Some(1));
    }
}
