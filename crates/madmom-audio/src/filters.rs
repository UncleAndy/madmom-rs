//! Frequency conversions and filterbanks.
//!
//! Port of `madmom/audio/filters.py` (1616 lines, 97 tests in
//! `tests/test_audio_filters.py`). Notes that decide the numbers:
//!
//! * `hz2bark`/`bark2hz` **raise** `NotImplementedError` upstream ("it produces
//!   negative values"). They are refused here as well — implementing them would
//!   invent behaviour the reference deliberately does not have. The Bark *filterbank*
//!   uses the hard-coded `bark_frequencies` table instead, which is ported.
//! * filters are normalized **in float32**: madmom casts the filter data to
//!   `float32` first and then divides by the sum of that float32 buffer;
//! * `TriangularFilter`'s falling edge is `linspace(1, 0, n, endpoint=False)`, so the
//!   last coefficient is `(n-1)/n` and never exactly zero;
//! * `frequencies2bins` picks the *closer* of the two neighbouring bin frequencies and
//!   keeps the upper one on a tie (`indices -= freq - left < right - freq`);
//! * a filterbank is `maximum`-combined per band (not summed) when several filters land
//!   in one band, and filters reaching outside the bin range are truncated.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use thiserror::Error;

/// Filter coefficients are stored as `float32` (`FILTER_DTYPE` upstream).
pub type FilterValue = f32;

/// Tuning frequency of A4 [Hz] (`A4` upstream).
pub const A4: f64 = 440.0;
/// Default minimum frequency of the filterbanks [Hz].
pub const FMIN: f64 = 30.0;
/// Default maximum frequency of the filterbanks [Hz].
pub const FMAX: f64 = 17000.0;
/// Default number of bands.
pub const NUM_BANDS: usize = 12;
/// Filters are normalized by default.
pub const NORM_FILTERS: bool = true;
/// Duplicate bins are removed by default.
pub const UNIQUE_FILTERS: bool = true;

/// Errors of the filter stage.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum FilterError {
    /// The reference implementation refuses this function; so do we.
    #[error("{0}")]
    NotImplemented(&'static str),
    /// A triangular filter needs `start <= center < stop`.
    #[error("`center` must be between `start` and `stop`")]
    CenterOutOfRange,
    /// A rectangular filter needs `start < stop`.
    #[error("`start` must be smaller than `stop`")]
    StartNotBeforeStop,
    /// Not enough bins were given to create the requested filters.
    #[error("not enough bins to create a {0}: got {1}")]
    NotEnoughBins(&'static str, usize),
    /// Overlapping rectangular filters are not implemented upstream either.
    #[error("overlapping RectangularFilters are not implemented")]
    OverlappingRectangularFilters,
    /// The number of bin frequencies must match the filterbank's first dimension.
    #[error("bin_frequencies has {found} entries, but the filterbank has {expected} rows")]
    BinFrequencyCountMismatch {
        /// Number of rows in the filterbank.
        expected: usize,
        /// Number of bin frequencies that were passed.
        found: usize,
    },
}

/// Converts Hz to Mel (`1127.01048 * ln(f / 700 + 1)`).
pub fn hz2mel(f: f64) -> f64 {
    1127.01048 * (f / 700.0 + 1.0).ln()
}

/// Converts Mel to Hz (`700 * (exp(m / 1127.01048) - 1)`).
pub fn mel2hz(m: f64) -> f64 {
    700.0 * ((m / 1127.01048).exp() - 1.0)
}

/// `num_bands` frequencies evenly spaced on the Mel scale between `fmin` and `fmax`.
pub fn mel_frequencies(num_bands: usize, fmin: f64, fmax: f64) -> Array1<f64> {
    let (low, high) = (hz2mel(fmin), hz2mel(fmax));
    if num_bands == 1 {
        return Array1::from_elem(1, mel2hz(low));
    }
    // np.linspace(low, high, num_bands) uses (high-low)/(n-1) as step
    let step = (high - low) / (num_bands - 1) as f64;
    (0..num_bands)
        .map(|index| mel2hz(low + step * index as f64))
        .collect()
}

/// Hz to Bark.
///
/// Refused exactly like the reference, which raises `NotImplementedError` because the
/// formula "produces negative values".
pub fn hz2bark(_f: f64) -> Result<f64, FilterError> {
    Err(FilterError::NotImplemented(
        "hz2bark is not implemented in the reference (it produces negative values)",
    ))
}

/// Bark to Hz — refused for the same reason as [`hz2bark`].
pub fn bark2hz(_z: f64) -> Result<f64, FilterError> {
    Err(FilterError::NotImplemented(
        "bark2hz is not implemented in the reference (it produces weird values)",
    ))
}

/// Frequencies of the Bark scale between `fmin` and `fmax` (the reference's fixed table).
pub fn bark_frequencies(fmin: f64, fmax: f64) -> Array1<f64> {
    const BARK: [f64; 25] = [
        20.0, 100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0,
        1720.0, 2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0,
        12000.0, 15500.0,
    ];
    searchsorted_slice(&BARK, fmin, fmax)
}

/// Frequencies of the doubled Bark scale (`bark_double_frequencies` table).
///
/// Transcribed from the reference, and pinned by `tests/golden/filters` — the first
/// version of this table had `60` where madmom has `50`, which the dump caught.
pub fn bark_double_frequencies(fmin: f64, fmax: f64) -> Array1<f64> {
    const BARK_DOUBLE: [f64; 49] = [
        20.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 510.0, 570.0, 630.0,
        700.0, 770.0, 840.0, 920.0, 1000.0, 1080.0, 1170.0, 1270.0, 1370.0, 1480.0, 1600.0, 1720.0,
        1850.0, 2000.0, 2150.0, 2320.0, 2500.0, 2700.0, 2900.0, 3150.0, 3400.0, 3700.0, 4000.0,
        4400.0, 4800.0, 5300.0, 5800.0, 6400.0, 7000.0, 7700.0, 8500.0, 9500.0, 10500.0, 12000.0,
        13500.0, 15500.0,
    ];
    searchsorted_slice(&BARK_DOUBLE, fmin, fmax)
}

/// Slice a sorted frequency table the way the reference does:
/// from the first entry `>= fmin` to the first entry `> fmax`.
fn searchsorted_slice(table: &[f64], fmin: f64, fmax: f64) -> Array1<f64> {
    let start = table.partition_point(|value| *value < fmin);
    let end = table.partition_point(|value| *value <= fmax);
    let end = end.max(start);
    Array1::from(table[start..end].to_vec())
}

/// Frequencies aligned on a logarithmic scale (`bands_per_octave` per octave).
///
/// `floor`/`ceil` widen the range first, then the frequencies are filtered to
/// `[fmin, fmax]` with `searchsorted`: lower bound inclusive, upper bound inclusive.
pub fn log_frequencies(
    bands_per_octave: usize,
    fmin: f64,
    fmax: f64,
    fref: f64,
) -> Result<Array1<f64>, FilterError> {
    if bands_per_octave == 0 {
        return Err(FilterError::NotImplemented(
            "log_frequencies needs at least one band per octave",
        ));
    }
    let left = ((fmin / fref).log2() * bands_per_octave as f64).floor() as i64;
    let right = ((fmax / fref).log2() * bands_per_octave as f64).ceil() as i64;
    let step = 1.0 / bands_per_octave as f64;
    let all: Vec<f64> = (left..right)
        .map(|index| fref * 2f64.powf(index as f64 * step))
        .collect();
    let start = all.partition_point(|value| *value < fmin);
    let end = all.partition_point(|value| *value <= fmax);
    let end = end.max(start);
    Ok(Array1::from(all[start..end].to_vec()))
}

/// Frequencies separated by semitones (`log_frequencies` with 12 bands per octave).
pub fn semitone_frequencies(fmin: f64, fmax: f64, fref: f64) -> Result<Array1<f64>, FilterError> {
    log_frequencies(12, fmin, fmax, fref)
}

/// Converts frequencies to MIDI notes (`12 * log2(f / fref) + 69`).
pub fn hz2midi(f: f64, fref: f64) -> f64 {
    12.0 * (f / fref).log2() + 69.0
}

/// Converts MIDI notes to frequencies (`2^((m - 69)/12) * fref`).
pub fn midi2hz(m: f64, fref: f64) -> f64 {
    2f64.powf((m - 69.0) / 12.0) * fref
}

/// Converts Hz to ERB (`21.4 * log10(1 + 4.37 * f / 1000)`).
pub fn hz2erb(f: f64) -> f64 {
    21.4 * (1.0 + 4.37 * f / 1000.0).log10()
}

/// Converts ERB to Hz (`(10^(e / 21.4) - 1) * 1000 / 4.37`).
pub fn erb2hz(e: f64) -> f64 {
    (10f64.powf(e / 21.4) - 1.0) * 1000.0 / 4.37
}

/// Maps frequencies to the indices of the closest bin frequencies.
///
/// The bins are clipped to `1..len-1` first (as the reference does), then the closer of
/// the two neighbouring frequencies wins; on a tie the upper bin is kept. With
/// `unique_bins` the result is deduplicated, which is what keeps low frequencies from
/// dominating a sum over bands.
pub fn frequencies2bins(
    frequencies: ArrayView1<'_, f64>,
    bin_frequencies: ArrayView1<'_, f64>,
    unique_bins: bool,
) -> Vec<usize> {
    let count = bin_frequencies.len();
    if count == 0 {
        return Vec::new();
    }
    // copy to a slice: `searchsorted` needs random access, a view can be strided
    let bin_frequencies = bin_frequencies.to_vec();
    let mut bins: Vec<usize> = Vec::with_capacity(frequencies.len());
    for &frequency in frequencies.iter() {
        let mut index = bin_frequencies.partition_point(|value| *value < frequency);
        index = index.clamp(1, count - 1);
        let left = bin_frequencies[index - 1];
        let right = bin_frequencies[index];
        if frequency - left < right - frequency {
            index -= 1;
        }
        bins.push(index);
    }
    if unique_bins {
        bins.sort_unstable();
        bins.dedup();
    }
    bins
}

/// Maps bin indices to their frequencies.
pub fn bins2frequencies(
    bins: &[usize],
    bin_frequencies: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, FilterError> {
    let mut out = Array1::zeros(bins.len());
    for (index, bin) in bins.iter().enumerate() {
        out[index] =
            *bin_frequencies
                .get(*bin)
                .ok_or_else(|| FilterError::BinFrequencyCountMismatch {
                    expected: bin_frequencies.len(),
                    found: *bin,
                })?;
    }
    Ok(out)
}

/// A filter: coefficients plus the bin it starts at.
#[derive(Debug, Clone, PartialEq)]
pub struct Filter {
    data: Array1<FilterValue>,
    start: i64,
    center: Option<i64>,
}

impl Filter {
    /// Build a filter from coefficients, casting and normalizing like the reference.
    ///
    /// The normalization happens in `float32`, after the cast, exactly as madmom's
    /// `Filter.__new__` does it (`obj /= np.sum(obj)` on a float32 buffer).
    pub fn new(data: Array1<FilterValue>, start: i64, norm: bool) -> Self {
        let mut data = data;
        if norm {
            let sum: f32 = data.iter().sum();
            if sum != 0.0 {
                data.mapv_inplace(|value| value / sum);
            }
        }
        Self {
            data,
            start,
            center: None,
        }
    }

    /// Coefficients.
    pub fn data(&self) -> ArrayView1<'_, FilterValue> {
        self.data.view()
    }

    /// First bin the filter covers.
    pub fn start(&self) -> i64 {
        self.start
    }

    /// One past the last bin the filter covers.
    pub fn stop(&self) -> i64 {
        self.start + self.len() as i64
    }

    /// Center bin, for filters that have one.
    pub fn center(&self) -> Option<i64> {
        self.center
    }

    /// Number of coefficients.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the filter has no coefficients at all.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// A triangular filter between `start` and `stop`, peaking at `center`.
pub fn triangular_filter(
    start: i64,
    center: i64,
    stop: i64,
    norm: bool,
) -> Result<Filter, FilterError> {
    if !(start <= center && center < stop) {
        return Err(FilterError::CenterOutOfRange);
    }
    let center = (center - start) as usize;
    let stop = (stop - start) as usize;
    let mut data = Array1::<FilterValue>::zeros(stop);
    // rising edge, without the center: linspace(0, 1, center, endpoint=False)
    if center > 0 {
        for index in 0..center {
            data[index] = (index as f64 / center as f64) as FilterValue;
        }
    }
    // falling edge, including the center: linspace(1, 0, stop - center, endpoint=False)
    let tail = stop - center;
    if tail > 0 {
        for index in 0..tail {
            data[center + index] = (1.0 - index as f64 / tail as f64) as FilterValue;
        }
    }
    let mut filter = Filter::new(data, start, norm);
    filter.center = Some(start + center as i64);
    Ok(filter)
}

/// A rectangular filter covering `[start, stop)`.
pub fn rectangular_filter(start: i64, stop: i64, norm: bool) -> Result<Filter, FilterError> {
    if start >= stop {
        return Err(FilterError::StartNotBeforeStop);
    }
    let data = Array1::<FilterValue>::ones((stop - start) as usize);
    Ok(Filter::new(data, start, norm))
}

/// Yields `(start, center, stop)` triples for triangular filters.
///
/// The windows slide by one bin, so consecutive filters overlap; with `overlap=false`
/// the edges are moved to the midpoints of the neighbouring centers (`floor`/`ceil`).
/// Filters narrower than two bins collapse to a single-bin peak.
pub fn triangular_band_bins(
    bins: &[usize],
    overlap: bool,
) -> Result<Vec<(usize, usize, usize)>, FilterError> {
    if bins.len() < 3 {
        return Err(FilterError::NotEnoughBins("TriangularFilter", bins.len()));
    }
    let mut out = Vec::new();
    let mut index = 0usize;
    while index + 3 <= bins.len() {
        let (mut start, mut center, mut stop) = (
            bins[index] as i64,
            bins[index + 1] as i64,
            bins[index + 2] as i64,
        );
        if !overlap {
            start = ((center + start) as f64 / 2.0).floor() as i64;
            stop = ((center + stop) as f64 / 2.0).ceil() as i64;
        }
        if stop - start < 2 {
            center = start;
            stop = start + 1;
        }
        out.push((start as usize, center as usize, stop as usize));
        index += 1;
    }
    Ok(out)
}

/// Yields `(start, stop)` pairs for rectangular filters.
pub fn rectangular_band_bins(
    bins: &[usize],
    overlap: bool,
) -> Result<Vec<(usize, usize)>, FilterError> {
    if bins.len() < 2 {
        return Err(FilterError::NotEnoughBins("RectangularFilter", bins.len()));
    }
    if overlap {
        return Err(FilterError::OverlappingRectangularFilters);
    }
    let mut out = Vec::new();
    let mut index = 0usize;
    while index + 2 <= bins.len() {
        out.push((bins[index], bins[index + 1]));
        index += 1;
    }
    Ok(out)
}

/// A filterbank: one column per band, one row per bin.
#[derive(Debug, Clone, PartialEq)]
pub struct Filterbank {
    data: Array2<FilterValue>,
    bin_frequencies: Array1<f64>,
}

impl Filterbank {
    /// Combine filters into a filterbank.
    ///
    /// Each band may hold several filters; where they overlap, the **maximum** wins (the
    /// reference merges them with `np.maximum`, it does not sum them), and filters
    /// reaching outside the bin range are truncated.
    pub fn from_filters(
        filters: &[Vec<Filter>],
        bin_frequencies: ArrayView1<'_, f64>,
    ) -> Result<Self, FilterError> {
        let rows = bin_frequencies.len();
        let mut data = Array2::<FilterValue>::zeros((rows, filters.len()));
        for (band_id, band_filters) in filters.iter().enumerate() {
            let mut band = data.column_mut(band_id);
            for filter in band_filters {
                put_filter(filter, &mut band);
            }
        }
        Self::new(data, bin_frequencies)
    }

    /// Build a filterbank from an existing matrix.
    pub fn new(
        data: Array2<FilterValue>,
        bin_frequencies: ArrayView1<'_, f64>,
    ) -> Result<Self, FilterError> {
        if data.shape()[0] != bin_frequencies.len() {
            return Err(FilterError::BinFrequencyCountMismatch {
                expected: data.shape()[0],
                found: bin_frequencies.len(),
            });
        }
        Ok(Self {
            data,
            bin_frequencies: bin_frequencies.to_owned(),
        })
    }

    /// The filterbank matrix, `(num_bins, num_bands)`.
    pub fn matrix(&self) -> ArrayView2<'_, FilterValue> {
        self.data.view()
    }

    /// Frequencies of the bins.
    pub fn bin_frequencies(&self) -> ArrayView1<'_, f64> {
        self.bin_frequencies.view()
    }

    /// Number of bins (rows).
    pub fn num_bins(&self) -> usize {
        self.data.shape()[0]
    }

    /// Number of bands (columns).
    pub fn num_bands(&self) -> usize {
        self.data.shape()[1]
    }

    /// Corner frequencies of every band, `(num_bands, 2)`.
    ///
    /// The lowest and highest non-zero bin of each band, mapped to their frequencies.
    pub fn corner_frequencies(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.num_bands(), 2));
        for band in 0..self.num_bands() {
            let (min_bin, max_bin) = self.band_extent(band);
            out[[band, 0]] = self.bin_frequencies[min_bin];
            out[[band, 1]] = self.bin_frequencies[max_bin];
        }
        out
    }

    /// Center frequencies of every band.
    ///
    /// For a band with a flat top (a rectangular filter) the middle bin is used, for a
    /// band with a peak the peak bin is — the slice excluding the last bin, exactly as
    /// the reference indexes it.
    pub fn center_frequencies(&self) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.num_bands());
        for band in 0..self.num_bands() {
            let (min_bin, max_bin) = self.band_extent(band);
            let column = self.data.column(band);
            let center = if column[min_bin] == column[max_bin] {
                min_bin + (max_bin - min_bin) / 2
            } else {
                let slice = column.slice(ndarray::s![min_bin..max_bin]);
                let peak = slice
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |acc, (index, value)| {
                        if *value > acc.1 {
                            (index, *value)
                        } else {
                            acc
                        }
                    })
                    .0;
                min_bin + peak
            };
            out[band] = self.bin_frequencies[center];
        }
        out
    }

    /// Lowest and highest non-zero bin of one band.
    fn band_extent(&self, band: usize) -> (usize, usize) {
        let column = self.data.column(band);
        let first = column
            .iter()
            .position(|value| *value != 0.0)
            .unwrap_or_else(|| panic!("band {band} has no non-zero bin"));
        let last = column
            .iter()
            .rposition(|value| *value != 0.0)
            .unwrap_or(first);
        (first, last)
    }

    /// Lowest frequency covered by the filterbank.
    pub fn fmin(&self) -> f64 {
        self.bin_frequencies[self
            .data
            .iter()
            .position(|value| *value != 0.0)
            .expect("empty filterbank")]
    }

    /// Highest frequency covered by the filterbank.
    pub fn fmax(&self) -> f64 {
        let index = self
            .data
            .rows()
            .into_iter()
            .rposition(|row| row.iter().any(|value| *value != 0.0))
            .expect("empty filterbank");
        self.bin_frequencies[index]
    }

    /// Apply the filterbank to a spectrogram slice: `spectrum · filterbank`.
    pub fn apply(&self, spectrum: ArrayView1<'_, f32>) -> Array1<f32> {
        assert_eq!(
            spectrum.len(),
            self.num_bins(),
            "spectrum has {} bins, the filterbank expects {}",
            spectrum.len(),
            self.num_bins()
        );
        let mut out = Array1::<f32>::zeros(self.num_bands());
        for band in 0..self.num_bands() {
            let mut acc = 0.0f32;
            for bin in 0..self.num_bins() {
                acc += spectrum[bin] * self.data[[bin, band]];
            }
            out[band] = acc;
        }
        out
    }
}

/// Puts one filter into a band column: truncate, then `maximum` with what is there.
fn put_filter(filter: &Filter, band: &mut ndarray::ArrayViewMut1<'_, FilterValue>) {
    let band_len = band.len() as i64;
    let mut data = filter.data().to_owned();
    let mut start = filter.start();
    let mut stop = start + data.len() as i64;
    if start < 0 {
        data = data.slice(ndarray::s![-start as usize..]).to_owned();
        start = 0;
    }
    if stop > band_len {
        let keep = (band_len - start).max(0) as usize;
        data = data.slice(ndarray::s![..keep]).to_owned();
        stop = band_len;
    }
    if stop <= start {
        return;
    }
    for (offset, value) in data.iter().enumerate() {
        let position = (start as usize) + offset;
        if position >= band.len() {
            break;
        }
        if *value > band[position] {
            band[position] = *value;
        }
    }
}

/// A Mel filterbank: `num_bands + 2` Mel-spaced frequencies as triangular filters.
pub fn mel_filterbank(
    bin_frequencies: ArrayView1<'_, f64>,
    num_bands: usize,
    fmin: f64,
    fmax: f64,
    norm_filters: bool,
    unique_filters: bool,
) -> Result<Filterbank, FilterError> {
    let frequencies = mel_frequencies(num_bands + 2, fmin, fmax);
    let bins = frequencies2bins(frequencies.view(), bin_frequencies, unique_filters);
    let filters = triangular_filters(&bins, norm_filters, true)?;
    Filterbank::from_filters(&filters, bin_frequencies)
}

/// A filterbank with roughly logarithmic band spacing (one triangular filter per band).
pub fn logarithmic_filterbank(
    bin_frequencies: ArrayView1<'_, f64>,
    bands_per_octave: usize,
    fmin: f64,
    fmax: f64,
    fref: f64,
    norm_filters: bool,
    unique_filters: bool,
) -> Result<Filterbank, FilterError> {
    let frequencies = log_frequencies(bands_per_octave, fmin, fmax, fref)?;
    let bins = frequencies2bins(frequencies.view(), bin_frequencies, unique_filters);
    let filters = triangular_filters(&bins, norm_filters, true)?;
    Filterbank::from_filters(&filters, bin_frequencies)
}

/// Spacing of a Bark filterbank (`num_bands` is the string `'normal'`/`'double'`
/// upstream, which is why it is an enum here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarkSpacing {
    /// The 25-entry Bark table (`bark_frequencies`).
    Normal,
    /// The denser 49-entry table (`bark_double_frequencies`).
    Double,
}

/// A Bark filterbank, built from the reference's fixed Bark frequency table.
pub fn bark_filterbank(
    bin_frequencies: ArrayView1<'_, f64>,
    spacing: BarkSpacing,
    fmin: f64,
    fmax: f64,
    norm_filters: bool,
    unique_filters: bool,
) -> Result<Filterbank, FilterError> {
    let frequencies = match spacing {
        BarkSpacing::Normal => bark_frequencies(fmin, fmax),
        BarkSpacing::Double => bark_double_frequencies(fmin, fmax),
    };
    // Two differences from the other filterbanks, straight from the reference:
    // `unique_filters` is **inverted** here (`unique_bins=not unique_filters`), and the
    // bands are built from non-overlapping **rectangular** filters, not triangular ones.
    let bins = frequencies2bins(frequencies.view(), bin_frequencies, !unique_filters);
    let filters = rectangular_filters(&bins, norm_filters, false)?;
    Filterbank::from_filters(&filters, bin_frequencies)
}

/// Turn bin pairs into rectangular filters, one `Vec` per filter (one band each).
fn rectangular_filters(
    bins: &[usize],
    norm: bool,
    overlap: bool,
) -> Result<Vec<Vec<Filter>>, FilterError> {
    let mut out = Vec::new();
    for (start, stop) in rectangular_band_bins(bins, overlap)? {
        out.push(vec![rectangular_filter(start as i64, stop as i64, norm)?]);
    }
    Ok(out)
}

/// Turn bin triples into triangular filters, one `Vec` per filter (one band each).
fn triangular_filters(
    bins: &[usize],
    norm: bool,
    overlap: bool,
) -> Result<Vec<Vec<Filter>>, FilterError> {
    let mut out = Vec::new();
    for (start, center, stop) in triangular_band_bins(bins, overlap)? {
        out.push(vec![triangular_filter(
            start as i64,
            center as i64,
            stop as i64,
            norm,
        )?]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn frequency_conversions_round_trip() {
        // 440 Hz is MIDI 69 and A4 by definition
        assert!((hz2midi(A4, A4) - 69.0).abs() < 1e-12);
        assert!((midi2hz(69.0, A4) - A4).abs() < 1e-12);
        // the Mel scale and the ERB scale are both invertible
        for frequency in [30.0, 440.0, 17000.0] {
            assert!((mel2hz(hz2mel(frequency)) - frequency).abs() < 1e-9);
            assert!((erb2hz(hz2erb(frequency)) - frequency).abs() < 1e-9);
        }
    }

    #[test]
    fn bark_conversions_are_refused_like_the_reference() {
        assert!(matches!(
            hz2bark(440.0),
            Err(FilterError::NotImplemented(_))
        ));
        assert!(matches!(bark2hz(3.0), Err(FilterError::NotImplemented(_))));
    }

    #[test]
    fn frequencies2bins_matches_the_reference() {
        // madmom: frequencies2bins([4, 7, 5, 16], [0, 10, 20, 30]) == [0, 1, 1, 2]
        // (the index is clipped to >= 1 *before* moving to the closer neighbour, so 4 Hz
        // still lands on bin 0; a tie keeps the upper bin)
        let bins = array![0.0f64, 10.0, 20.0, 30.0];
        let frequencies = array![4.0f64, 7.0, 5.0, 16.0];
        assert_eq!(
            frequencies2bins(frequencies.view(), bins.view(), false),
            vec![0, 1, 1, 2]
        );
        // duplicates are removed on request
        let many = array![7.0f64, 8.0, 9.0];
        assert_eq!(
            frequencies2bins(many.view(), bins.view(), false),
            vec![1, 1, 1]
        );
        assert_eq!(frequencies2bins(many.view(), bins.view(), true), vec![1]);
    }

    #[test]
    fn triangular_filter_has_the_reference_edges() {
        // madmom: TriangularFilter(0, 2, 4) == [0.0, 0.5, 1.0, 0.5] — the falling edge
        // is linspace(1, 0, 2, endpoint=False), so it never reaches zero
        let filter = triangular_filter(0, 2, 4, false).unwrap();
        assert_eq!(filter.data().to_vec(), vec![0.0, 0.5, 1.0, 0.5]);
        assert_eq!(filter.center(), Some(2));
        assert_eq!(filter.stop(), 4);
        assert!(matches!(
            triangular_filter(0, 5, 4, false),
            Err(FilterError::CenterOutOfRange)
        ));
    }

    #[test]
    fn normalization_makes_the_coefficients_sum_to_one() {
        let filter = triangular_filter(0, 2, 4, true).unwrap();
        let sum: f32 = filter.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn narrow_filters_collapse_to_a_single_bin() {
        // madmom: TriangularFilter.band_bins([1, 2, 2]) == [(1, 1, 2)]
        let bins = vec![1usize, 2, 2];
        let triples = triangular_band_bins(&bins, true).unwrap();
        assert_eq!(triples, vec![(1, 1, 2)]);
        // start == stop would be invalid, so the helper moves the stop by one
        let filters = triangular_filters(&bins, false, true).unwrap();
        assert_eq!(filters.len(), 1);
    }

    #[test]
    fn filterbank_merges_overlapping_filters_by_maximum() {
        // two filters in one band: the maximum, not the sum, lands in the band
        let bins = array![0.0f64, 1.0, 2.0];
        let band = vec![
            triangular_filter(0, 1, 2, false).unwrap(),
            triangular_filter(1, 2, 3, false).unwrap(),
        ];
        let filterbank = Filterbank::from_filters(&[band, vec![]], bins.view()).unwrap();
        assert_eq!(filterbank.num_bins(), 3);
        assert_eq!(filterbank.num_bands(), 2);
        let column = filterbank.matrix().column(0).to_owned();
        assert_eq!(column.to_vec(), vec![0.0, 1.0, 1.0]);
    }

    #[test]
    fn logarithmic_filterbank_spacing_and_band_count() {
        let bins: Vec<f64> = (0..1024).map(|bin| bin as f64 * 21.53).collect();
        let bins = Array1::from(bins);
        let filterbank =
            logarithmic_filterbank(bins.view(), 12, 30.0, 17000.0, A4, true, true).unwrap();
        // madmom: LogarithmicFilterbank(bf, 12, 30, 17000, 440, True, True) has 81 bands
        // (110 log frequencies, but duplicate bins are removed), 108 without deduplication
        assert_eq!(filterbank.num_bands(), 81);
        let all = logarithmic_filterbank(bins.view(), 12, 30.0, 17000.0, A4, true, false).unwrap();
        assert_eq!(all.num_bands(), 108);
        // every band is normalized to sum 1 (within float32 rounding)
        for band in 0..filterbank.num_bands() {
            let sum: f32 = filterbank.matrix().column(band).iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "band {band} sums to {sum}");
        }
    }
}
