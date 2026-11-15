//! Feed forward and feed backward comb filters and their filter bank.
//!
//! Port of `madmom/audio/comb_filters.py` (the reference is Cython). A comb filter adds a
//! delayed copy of the signal — of the *input* for the feed forward variant
//! (`y[n] = x[n] + α * x[n - τ]`), of the *output* for the feed backward one
//! (`y[n] = x[n] + α * y[n - τ]`). The backward filter resonates at a period of `τ`
//! frames, which is what turns a beat activation function into an inter-beat-interval
//! histogram: a bank of them, one per candidate interval, is fed the same activations and
//! the bin whose delay matches the pulse keeps the largest response.
//!
//! Everything is `f64`: the tempo module casts the activations to float before filtering
//! (`astype(float)`), so the f32 pipeline used for signals and spectrograms would lose the
//! last digits of the histogram.

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};
use thiserror::Error;

/// Errors of the comb filter functions.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CombFilterError {
    /// `tau` must be greater than zero (both reference implementations raise here).
    #[error("`tau` must be greater than 0, got {0}")]
    InvalidTau(usize),
    /// `tau` and `alpha` must have the same length.
    #[error("`tau` and `alpha` must have the same length: {tau} against {alpha}")]
    LengthMismatch {
        /// Number of delays.
        tau: usize,
        /// Number of scaling factors.
        alpha: usize,
    },
    /// At least one delay is required.
    #[error("at least one tau is required")]
    NoDelays,
    /// The signal must be 1-D or 2-D.
    #[error("signal must be 1d or 2d, got {0} dimensions")]
    InvalidDims(usize),
}

/// Feed forward comb filter: `y[n] = x[n] + α * x[n - τ]`.
///
/// The delayed term comes from the **input**, not from the running output, so the result
/// is a plain two-tap FIR filter.
pub fn feed_forward_comb_filter(
    signal: ArrayView1<'_, f64>,
    tau: usize,
    alpha: f64,
) -> Result<Array1<f64>, CombFilterError> {
    if tau == 0 {
        return Err(CombFilterError::InvalidTau(tau));
    }
    let mut y = signal.to_owned();
    let len = signal.len();
    if tau < len {
        for n in tau..len {
            y[n] += alpha * signal[n - tau];
        }
    }
    Ok(y)
}

/// Feed backward comb filter: `y[n] = x[n] + α * y[n - τ]`.
///
/// This one is recursive: the delayed term is the already filtered output, so the filter
/// resonates at the period `τ`.
///
/// **`alpha` is rounded to `f32`** before use, because the reference's Cython helper
/// declares it as a C `float` (`_feed_backward_comb_filter_1d(..., float alpha)`) while the
/// signal stays `float64`. The forward filter is pure numpy and keeps `f64`. Skipping this
/// rounding costs ~4e-7 on a comb histogram of the reference's own activations — visible,
/// because the tempo strengths are compared to 1e-3.
pub fn feed_backward_comb_filter(
    signal: ArrayView1<'_, f64>,
    tau: usize,
    alpha: f64,
) -> Result<Array1<f64>, CombFilterError> {
    if tau == 0 {
        return Err(CombFilterError::InvalidTau(tau));
    }
    let alpha = alpha as f32 as f64;
    let mut y = signal.to_owned();
    let len = signal.len();
    if tau < len {
        for n in tau..len {
            y[n] += alpha * y[n - tau];
        }
    }
    Ok(y)
}

/// The same recursion for a matrix, applied along the first axis of every column.
///
/// Note on parity: the reference's Cython `_feed_backward_comb_filter_2d` loops
/// `for d in range(2)`, i.e. it filters only the first two columns of a 2-D signal and
/// copies the rest through untouched. The port filters **all** columns; for one or two
/// columns — every input madmom's own pipeline produces — the results are identical, and
/// for more columns the reference silently leaves data unfiltered.
pub fn feed_backward_comb_filter_2d(
    signal: ArrayView2<'_, f64>,
    tau: usize,
    alpha: f64,
) -> Result<Array2<f64>, CombFilterError> {
    if tau == 0 {
        return Err(CombFilterError::InvalidTau(tau));
    }
    // the reference's 2-D helper declares `float alpha` as well
    let alpha = alpha as f32 as f64;
    let mut y = signal.to_owned();
    let (rows, cols) = (signal.shape()[0], signal.shape()[1]);
    if tau < rows {
        for column in 0..cols {
            for n in tau..rows {
                let delayed = y[[n - tau, column]];
                y[[n, column]] += alpha * delayed;
            }
        }
    }
    Ok(y)
}

/// Which comb filter a bank uses (madmom's `filter_function` literal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombFilterFunction {
    /// `'forward'` / `feed_forward_comb_filter`.
    Forward,
    /// `'backward'` / `feed_backward_comb_filter`.
    Backward,
}

impl CombFilterFunction {
    /// Filter a 1-D signal.
    pub fn filter_1d(
        self,
        signal: ArrayView1<'_, f64>,
        tau: usize,
        alpha: f64,
    ) -> Result<Array1<f64>, CombFilterError> {
        match self {
            CombFilterFunction::Forward => feed_forward_comb_filter(signal, tau, alpha),
            CombFilterFunction::Backward => feed_backward_comb_filter(signal, tau, alpha),
        }
    }
}

/// Filter a 1-D signal with a bank of comb filters, one per delay.
///
/// Returns a `(signal length, number of delays)` matrix — the different taus are aligned
/// along the (new) last dimension, like the reference's `np.vstack(...).T`.
pub fn comb_filter(
    signal: ArrayView1<'_, f64>,
    function: CombFilterFunction,
    tau: &[usize],
    alpha: &[f64],
) -> Result<Array2<f64>, CombFilterError> {
    if tau.is_empty() {
        return Err(CombFilterError::NoDelays);
    }
    if tau.len() != alpha.len() {
        return Err(CombFilterError::LengthMismatch {
            tau: tau.len(),
            alpha: alpha.len(),
        });
    }
    let mut out = Array2::<f64>::zeros((signal.len(), tau.len()));
    for (index, (tau, alpha)) in tau.iter().zip(alpha.iter()).enumerate() {
        let filtered = function.filter_1d(signal, *tau, *alpha)?;
        for (row, value) in filtered.iter().enumerate() {
            out[[row, index]] = *value;
        }
    }
    Ok(out)
}

/// The 2-D variant: a `(rows, columns, delays)` stack (the reference's `np.dstack`).
pub fn comb_filter_2d(
    signal: ArrayView2<'_, f64>,
    function: CombFilterFunction,
    tau: &[usize],
    alpha: &[f64],
) -> Result<Array3<f64>, CombFilterError> {
    if tau.is_empty() {
        return Err(CombFilterError::NoDelays);
    }
    if tau.len() != alpha.len() {
        return Err(CombFilterError::LengthMismatch {
            tau: tau.len(),
            alpha: alpha.len(),
        });
    }
    let (rows, cols) = (signal.shape()[0], signal.shape()[1]);
    let mut out = Array3::<f64>::zeros((rows, cols, tau.len()));
    for (index, (tau, alpha)) in tau.iter().zip(alpha.iter()).enumerate() {
        let filtered = match function {
            CombFilterFunction::Backward => feed_backward_comb_filter_2d(signal, *tau, *alpha)?,
            CombFilterFunction::Forward => {
                // the reference has no 2-D feed forward variant; apply it per column
                let mut y = signal.to_owned();
                if *tau < rows {
                    for column in 0..cols {
                        for n in *tau..rows {
                            y[[n, column]] += *alpha * signal[[n - *tau, column]];
                        }
                    }
                }
                y
            }
        };
        for row in 0..rows {
            for column in 0..cols {
                out[[row, column, index]] = filtered[[row, column]];
            }
        }
    }
    Ok(out)
}

/// A bank of comb filters, one per delay (madmom's `CombFilterbankProcessor`).
#[derive(Debug, Clone, PartialEq)]
pub struct CombFilterbankProcessor {
    function: CombFilterFunction,
    tau: Vec<usize>,
    alpha: Vec<f64>,
}

impl CombFilterbankProcessor {
    /// Build the bank; a single alpha is repeated for every delay, like the reference.
    pub fn new(
        function: CombFilterFunction,
        tau: Vec<usize>,
        alpha: Vec<f64>,
    ) -> Result<Self, CombFilterError> {
        if tau.is_empty() {
            return Err(CombFilterError::NoDelays);
        }
        let alpha = if alpha.len() == 1 {
            vec![alpha[0]; tau.len()]
        } else {
            alpha
        };
        if tau.len() != alpha.len() {
            return Err(CombFilterError::LengthMismatch {
                tau: tau.len(),
                alpha: alpha.len(),
            });
        }
        Ok(Self {
            function,
            tau,
            alpha,
        })
    }

    /// The filter function in use.
    pub fn function(&self) -> CombFilterFunction {
        self.function
    }

    /// The delays.
    pub fn tau(&self) -> &[usize] {
        &self.tau
    }

    /// The scaling factors.
    pub fn alpha(&self) -> &[f64] {
        &self.alpha
    }

    /// Filter a 1-D signal.
    pub fn process_1d(&self, signal: ArrayView1<'_, f64>) -> Result<Array2<f64>, CombFilterError> {
        comb_filter(signal, self.function, &self.tau, &self.alpha)
    }

    /// Filter a 2-D signal.
    pub fn process_2d(&self, signal: ArrayView2<'_, f64>) -> Result<Array3<f64>, CombFilterError> {
        comb_filter_2d(signal, self.function, &self.tau, &self.alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// The reference's own doctest values (`madmom/audio/comb_filters.py`).
    #[test]
    fn feed_forward_doctest() {
        let x = array![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let y = feed_forward_comb_filter(x.view(), 3, 0.5).unwrap();
        assert_eq!(
            y.to_vec(),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.5, 0.0, 0.0, 1.5]
        );
    }

    #[test]
    fn feed_backward_doctest() {
        let x = array![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let y = feed_backward_comb_filter(x.view(), 3, 0.5).unwrap();
        assert_eq!(
            y.to_vec(),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.5, 0.0, 0.0, 1.75]
        );
    }

    #[test]
    fn comb_filter_doctests() {
        let x = array![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let forward =
            comb_filter(x.view(), CombFilterFunction::Forward, &[2, 3], &[0.5, 0.5]).unwrap();
        assert_eq!(forward.shape(), &[9, 2]);
        assert_eq!(
            forward.row(8).to_vec(),
            vec![1.0, 1.5],
            "the last row is the filter's steady state"
        );
        assert_eq!(forward.row(4).to_vec(), vec![0.5, 0.0]);
        let backward =
            comb_filter(x.view(), CombFilterFunction::Backward, &[2, 3], &[0.5, 0.5]).unwrap();
        assert_eq!(backward.row(6).to_vec(), vec![0.25, 0.0]);
        assert_eq!(backward.row(8).to_vec(), vec![1.125, 1.75]);
    }

    #[test]
    fn a_single_alpha_is_repeated_and_tau_must_be_positive() {
        let bank =
            CombFilterbankProcessor::new(CombFilterFunction::Backward, vec![2, 3, 4], vec![0.79])
                .unwrap();
        assert_eq!(bank.alpha(), &[0.79, 0.79, 0.79]);
        let x = array![1.0, 0.0, 0.0, 0.0];
        assert_eq!(
            feed_backward_comb_filter(x.view(), 0, 0.5),
            Err(CombFilterError::InvalidTau(0))
        );
        assert_eq!(
            CombFilterbankProcessor::new(
                CombFilterFunction::Forward,
                vec![2, 3],
                vec![0.1, 0.2, 0.3]
            ),
            Err(CombFilterError::LengthMismatch { tau: 2, alpha: 3 })
        );
    }

    #[test]
    fn two_dimensional_backward_filter_filters_every_column() {
        // documents the deliberate divergence: the reference's Cython loops over two
        // columns only, this port filters all of them
        let signal = array![
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ];
        let y = feed_backward_comb_filter_2d(signal.view(), 1, 0.5).unwrap();
        assert_eq!(y.row(3).to_vec(), vec![0.125, 0.125, 0.125]);
    }
}
