//! The `scipy` behaviour the port depends on, reimplemented with scipy's exact semantics.
//!
//! Only what madmom actually calls lives here. Each function documents the reference
//! behaviour it copies, because the differences that matter are small (which side of a
//! comparison a plateau falls on, how the ends are treated) and invisible in the caller.

use thiserror::Error;

/// Errors of the `scipy` compatibility layer.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ScipyError {
    /// `order` must be at least 1, like `scipy.signal._boolrelextrema`.
    #[error("order must be an int >= 1, got {0}")]
    InvalidOrder(usize),
    /// The comparison needs at least one element.
    #[error("cannot find extrema of an empty array")]
    Empty,
}

/// How the shifted arrays are filled at the ends, i.e. numpy's `take(..., mode=...)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Out-of-range indices are clamped to the first/last element (`scipy`'s default).
    Clip,
    /// Indices wrap around (`mode='wrap'`).
    Wrap,
}

fn shifted(values: &[f64], index: i64, mode: Mode) -> f64 {
    let len = values.len() as i64;
    match mode {
        Mode::Wrap => values[index.rem_euclid(len) as usize],
        Mode::Clip => values[index.clamp(0, len - 1) as usize],
    }
}

/// `scipy.signal.argrelmax(values, order, mode)` for a 1-D array.
///
/// A value is a relative maximum when it is **strictly** greater than its `order`
/// neighbours on both sides; equal neighbours therefore disqualify a peak, which is what
/// makes plateaus disappear instead of producing one peak per element. With
/// [`Mode::Wrap`] the comparison at the ends uses the opposite end of the array, exactly
/// as `np.take(..., mode='wrap')` does.
pub fn argrelmax(values: &[f64], order: usize, mode: Mode) -> Result<Vec<usize>, ScipyError> {
    relative_extrema(values, order, mode, true)
}

/// `scipy.signal.argrelmin(values, order, mode)` for a 1-D array.
pub fn argrelmin(values: &[f64], order: usize, mode: Mode) -> Result<Vec<usize>, ScipyError> {
    relative_extrema(values, order, mode, false)
}

fn relative_extrema(
    values: &[f64],
    order: usize,
    mode: Mode,
    greater: bool,
) -> Result<Vec<usize>, ScipyError> {
    if order < 1 {
        return Err(ScipyError::InvalidOrder(order));
    }
    if values.is_empty() {
        return Err(ScipyError::Empty);
    }
    let mut out = Vec::new();
    for index in 0..values.len() as i64 {
        let main = values[index as usize];
        let mut is_extreme = true;
        for shift in 1..=order as i64 {
            let plus = shifted(values, index + shift, mode);
            let minus = shifted(values, index - shift, mode);
            // NaN fails both comparisons, so it is never an extremum — as in numpy
            let (left, right) = if greater {
                (main > minus, main > plus)
            } else {
                (main < minus, main < plus)
            };
            if !(left && right) {
                is_extreme = false;
                break;
            }
        }
        if is_extreme {
            out.push(index as usize);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_maxima_only() {
        // scipy: argrelmax([1, 2, 2, 1]) -> [] (the plateau has no strict maximum)
        assert!(argrelmax(&[1.0, 2.0, 2.0, 1.0], 1, Mode::Clip)
            .unwrap()
            .is_empty());
        assert_eq!(argrelmax(&[1.0, 2.0, 1.0], 1, Mode::Clip).unwrap(), vec![1]);
    }

    #[test]
    fn wrap_mode_uses_the_opposite_end() {
        // element 0 beats its wrapped neighbour (the last element), element 2 does not:
        // wrapped, its right neighbour is element 0 again
        assert_eq!(argrelmax(&[3.0, 1.0, 2.0], 1, Mode::Wrap).unwrap(), vec![0]);
        assert!(argrelmax(&[3.0, 1.0, 2.0], 1, Mode::Clip)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn minima_and_order() {
        assert_eq!(
            argrelmin(&[2.0, 1.0, 2.0, 1.0, 3.0], 1, Mode::Clip).unwrap(),
            vec![1, 3]
        );
        // order 2 requires a strictly lower value on both sides within two samples
        assert_eq!(
            argrelmin(&[9.0, 5.0, 4.0, 1.0, 8.0], 2, Mode::Clip).unwrap(),
            vec![3]
        );
        assert_eq!(
            argrelmax(&[1.0, 2.0, 1.0], 0, Mode::Clip),
            Err(ScipyError::InvalidOrder(0))
        );
    }
}
