//! Readers for NumPy `.npy` and `.npz` files.
//!
//! Deliberately free of C dependencies: the parity fixtures shipped with
//! madmom are read on every test run, so the reader must not drag in C
//! libraries or a compression stack. The observations this implements were
//! verified against the actual fixtures:
//!
//! - `.npy` files here are version 1.0 (`<f4`, `<i8`, `|O`) and little endian;
//! - `np.savez` writes **stored** (uncompressed) zip members, so a minimal zip
//!   reader is enough; deflated members are reported as an error instead of
//!   being silently mis-read;
//! - `fps` is a 0-d array, and in `sample.beats_tcn_tempo.npz` it is an object
//!   array (`|O`) because it was written by pickling. Object arrays are
//!   reported explicitly ([`NpyError::ObjectArray`]) so callers can degrade to
//!   "no frame rate" instead of guessing.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use num_complex::Complex32;

/// Result type used by this module.
pub type Result<T> = std::result::Result<T, NpyError>;

/// Errors produced while reading NumPy containers.
#[derive(Debug)]
pub enum NpyError {
    /// The file could not be read.
    Io {
        /// Path that was being read.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// The file does not start with the NumPy magic.
    BadMagic,
    /// The `.npy` format version is not supported.
    UnsupportedVersion {
        /// Major version byte of the container.
        major: u8,
        /// Minor version byte of the container.
        minor: u8,
    },
    /// The header dictionary could not be parsed.
    BadHeader(String),
    /// The buffer ended before the array data did.
    BufferTooShort {
        /// Number of bytes the header describes.
        need: usize,
        /// Number of bytes actually available.
        have: usize,
    },
    /// The dtype is not among the supported ones.
    UnsupportedDtype(String),
    /// The array has object dtype (`|O`), i.e. it holds pickled Python objects.
    ObjectArray,
    /// A zip member uses a compression method this reader does not implement.
    UnsupportedZipMethod(u16),
    /// The requested member does not exist in the zip container.
    MissingMember(String),
    /// A scalar was requested from an array with more than zero dimensions.
    NotScalar {
        /// Shape that was found instead of the expected 0-d array.
        shape: Vec<usize>,
    },
}

impl fmt::Display for NpyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NpyError::Io { path, source } => {
                write!(f, "cannot read {}: {}", path.display(), source)
            }
            NpyError::BadMagic => write!(f, "not a NumPy file (bad magic)"),
            NpyError::UnsupportedVersion { major, minor } => {
                write!(f, "unsupported .npy version {}.{}", major, minor)
            }
            NpyError::BadHeader(msg) => write!(f, "cannot parse .npy header: {}", msg),
            NpyError::BufferTooShort { need, have } => {
                write!(
                    f,
                    "array data truncated: need {} bytes, have {}",
                    need, have
                )
            }
            NpyError::UnsupportedDtype(d) => write!(f, "unsupported dtype: {}", d),
            NpyError::ObjectArray => write!(
                f,
                "object array (pickled Python objects); set fps explicitly instead"
            ),
            NpyError::UnsupportedZipMethod(m) => {
                write!(f, "unsupported zip compression method {}", m)
            }
            NpyError::MissingMember(name) => write!(f, "zip member {} not found", name),
            NpyError::NotScalar { shape } => write!(f, "expected 0-d array, got shape {:?}", shape),
        }
    }
}

impl std::error::Error for NpyError {}

/// Data type of a `.npy` array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// `<f4`
    F32,
    /// `<f8`
    F64,
    /// `<i2`
    I16,
    /// `<i4`
    I32,
    /// `<i8`
    I64,
    /// `|u1`
    U8,
    /// `<u2`
    U16,
    /// `<u4`
    U32,
    /// `<u8`
    U64,
    /// `<c8` — complex64 (what madmom's STFT dumps use).
    C64,
    /// `<c16` — complex128.
    C128,
    /// `|O` — pickled Python objects.
    Object,
}

impl DType {
    /// Parses a NumPy `descr` string such as `<f4`.
    pub fn from_descr(descr: &str) -> Result<Self> {
        Ok(match descr {
            "<f4" => DType::F32,
            "<f8" => DType::F64,
            "<i2" => DType::I16,
            "<i4" => DType::I32,
            "<i8" => DType::I64,
            "|u1" | "<u1" => DType::U8,
            "<u2" => DType::U16,
            "<u4" => DType::U32,
            "<u8" => DType::U64,
            "<c8" => DType::C64,
            "<c16" => DType::C128,
            "|O" => DType::Object,
            other => return Err(NpyError::UnsupportedDtype(other.to_string())),
        })
    }

    /// Size of one element in bytes.
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            // a complex element is *two* reals: complex64 is 8 bytes, not 4
            DType::F64 | DType::I64 | DType::U64 | DType::C64 => 8,
            DType::C128 => 16,
            DType::I16 | DType::U16 => 2,
            DType::U8 => 1,
            DType::Object => 0,
        }
    }

    /// True for the integer types.
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            DType::I16 | DType::I32 | DType::I64 | DType::U8 | DType::U16 | DType::U32 | DType::U64
        )
    }
}

/// Parsed `.npy` header plus the byte range holding the raw array data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// Element type.
    pub dtype: DType,
    /// `fortran_order` flag of the header.
    pub fortran_order: bool,
    /// Array shape; empty for scalars.
    pub shape: Vec<usize>,
    /// Offset of the first data byte inside the file.
    pub data_offset: usize,
}

impl Header {
    /// Number of elements described by the header.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }

    /// Number of bytes of array data.
    pub fn data_len(&self) -> usize {
        self.num_elements() * self.dtype.size_in_bytes()
    }
}

/// A `.npy` array: header plus its raw little-endian data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpyArray {
    /// Parsed header.
    pub header: Header,
    /// Raw array data (little endian, C order).
    pub data: Vec<u8>,
}

impl NpyArray {
    /// Element type.
    pub fn dtype(&self) -> DType {
        self.header.dtype
    }

    /// Shape; empty for scalars.
    pub fn shape(&self) -> &[usize] {
        &self.header.shape
    }

    /// Reinterprets the data as `f32`, converting from other numeric types.
    /// Reads the array as `f32` values.
    ///
    /// Every numeric dtype is converted (they are what audio arrives in: `.npy` dumps
    /// of int16 signals and float32 features alike); object arrays are refused, since
    /// a pickled Python object has no numeric value to fall back on.
    pub fn as_f32(&self) -> Result<Vec<f32>> {
        if self.header.dtype == DType::Object {
            return Err(NpyError::ObjectArray);
        }
        let out = match self.header.dtype {
            DType::F32 => self
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            DType::F64 => self
                .data
                .chunks_exact(8)
                .map(|c| {
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect(),
            DType::I32 => self
                .data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                .collect(),
            DType::I64 => self
                .data
                .chunks_exact(8)
                .map(|c| {
                    i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect(),
            DType::I16 => self
                .data
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32)
                .collect(),
            DType::U16 => self
                .data
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]) as f32)
                .collect(),
            DType::U32 => self
                .data
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                .collect(),
            DType::U64 => self
                .data
                .chunks_exact(8)
                .map(|c| {
                    u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect(),
            DType::U8 => self.data.iter().map(|b| *b as f32).collect(),
            DType::C64 | DType::C128 => {
                return Err(NpyError::UnsupportedDtype(format!(
                    "{:?} as f32 (use as_complex32)",
                    self.header.dtype
                )))
            }
            DType::Object => return Err(NpyError::ObjectArray),
        };
        Ok(out)
    }

    /// Reads the array as `f64` values, without the precision loss of [`Self::as_f32`].
    ///
    /// Needed for scalars that were written in double precision (log probabilities of a
    /// decoding, for instance), where `f32` rounding is coarser than the comparison.
    pub fn as_f64(&self) -> Result<Vec<f64>> {
        let out = match self.header.dtype {
            DType::F64 => self
                .data
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect(),
            DType::I64 => self
                .data
                .chunks_exact(8)
                .map(|c| {
                    i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f64
                })
                .collect(),
            DType::U64 => self
                .data
                .chunks_exact(8)
                .map(|c| {
                    u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f64
                })
                .collect(),
            DType::C64 | DType::C128 | DType::Object => {
                return Err(NpyError::UnsupportedDtype(format!(
                    "{:?} as f64",
                    self.header.dtype
                )))
            }
            other => self
                .as_f32_for(other)?
                .into_iter()
                .map(|value| value as f64)
                .collect(),
        };
        Ok(out)
    }

    /// `as_f32` for an explicit dtype (helper so `as_f64` can reuse the narrow types).
    fn as_f32_for(&self, dtype: DType) -> Result<Vec<f32>> {
        let clone = NpyArray {
            header: Header {
                dtype,
                ..self.header.clone()
            },
            data: self.data.clone(),
        };
        clone.as_f32()
    }

    /// Reads a complex array as `Complex32` values.
    ///
    /// The STFT dumps are `complex64` (`<c8`), so this is the dtype the audio stages
    /// compare against. Interleaved little-endian re/im pairs, as NumPy stores them.
    pub fn as_complex32(&self) -> Result<Vec<Complex32>> {
        let out = match self.header.dtype {
            DType::C64 => self
                .data
                .chunks_exact(8)
                .map(|c| {
                    let re = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    let im = f32::from_le_bytes([c[4], c[5], c[6], c[7]]);
                    Complex32::new(re, im)
                })
                .collect(),
            DType::C128 => self
                .data
                .chunks_exact(16)
                .map(|c| {
                    let re = f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
                    let im =
                        f64::from_le_bytes([c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15]]);
                    Complex32::new(re as f32, im as f32)
                })
                .collect(),
            other => {
                return Err(NpyError::UnsupportedDtype(format!(
                    "{other:?} as complex32"
                )))
            }
        };
        Ok(out)
    }

    /// Interprets a 0-d integer/float array as `f64`.
    pub fn scalar_f64(&self) -> Result<f64> {
        if !self.header.shape.is_empty() {
            return Err(NpyError::NotScalar {
                shape: self.header.shape.clone(),
            });
        }
        let bytes = &self.data;
        let value = match self.header.dtype {
            DType::F64 => f64::from_le_bytes(bytes[..8].try_into().unwrap()),
            DType::F32 => f32::from_le_bytes(bytes[..4].try_into().unwrap()) as f64,
            DType::I64 => i64::from_le_bytes(bytes[..8].try_into().unwrap()) as f64,
            DType::I32 => i32::from_le_bytes(bytes[..4].try_into().unwrap()) as f64,
            DType::U8 => bytes[0] as f64,
            DType::Object => return Err(NpyError::ObjectArray),
            other => return Err(NpyError::UnsupportedDtype(format!("{:?} as f64", other))),
        };
        Ok(value)
    }
}

/// Reads a `.npy` file from disk.
pub fn read_npy<P: AsRef<Path>>(path: P) -> Result<NpyArray> {
    let path = path.as_ref().to_path_buf();
    let bytes = fs::read(&path).map_err(|source| NpyError::Io {
        path: path.clone(),
        source,
    })?;
    parse_npy(&bytes)
}

/// Parses an in-memory `.npy` buffer.
pub fn parse_npy(bytes: &[u8]) -> Result<NpyArray> {
    let mut header = parse_header(bytes)?;
    let start = header.data_offset;
    let len = header.data_len();
    let end = start
        .checked_add(len)
        .ok_or(NpyError::BadHeader("size overflow".into()))?;
    if bytes.len() < end {
        return Err(NpyError::BufferTooShort {
            need: end,
            have: bytes.len(),
        });
    }
    let data = c_order_data(&header, &bytes[start..end])?;
    // the values are handed out in C order from here on
    header.fortran_order = false;
    Ok(NpyArray { header, data })
}

/// Reorders a Fortran-ordered array into C order.
///
/// `np.save` writes a transposed view (which is what madmom's processors return) with
/// `fortran_order: True`, so the raw bytes are column-major. Readers that ignore the flag
/// silently hand out a transposed array, which is exactly the kind of silent mismatch the
/// golden dumps are supposed to catch: the first (down-)beat then took its time from the
/// correct row but its beat number from the wrong column.
fn c_order_data(header: &Header, data: &[u8]) -> Result<Vec<u8>> {
    let shape = &header.shape;
    if !header.fortran_order || shape.len() < 2 {
        return Ok(data.to_vec());
    }
    let itemsize = header.dtype.size_in_bytes();
    let total: usize = shape.iter().product();
    if data.len() < total * itemsize {
        return Err(NpyError::BufferTooShort {
            need: total * itemsize,
            have: data.len(),
        });
    }
    // Fortran strides in elements: first axis varies fastest
    let mut fortran_strides = vec![1usize; shape.len()];
    for axis in 1..shape.len() {
        fortran_strides[axis] = fortran_strides[axis - 1] * shape[axis - 1];
    }
    let mut out = vec![0u8; total * itemsize];
    let mut index = vec![0usize; shape.len()];
    for c_index in 0..total {
        let mut remainder = c_index;
        for axis in (0..shape.len()).rev() {
            index[axis] = remainder % shape[axis];
            remainder /= shape[axis];
        }
        let f_index: usize = index
            .iter()
            .zip(fortran_strides.iter())
            .map(|(position, stride)| position * stride)
            .sum();
        out[c_index * itemsize..(c_index + 1) * itemsize]
            .copy_from_slice(&data[f_index * itemsize..(f_index + 1) * itemsize]);
    }
    Ok(out)
}

/// Parses the `.npy` magic, version and header dictionary.
pub fn parse_header(bytes: &[u8]) -> Result<Header> {
    if bytes.len() < 10 {
        return Err(NpyError::BufferTooShort {
            need: 10,
            have: bytes.len(),
        });
    }
    if &bytes[..6] != b"\x93NUMPY" {
        return Err(NpyError::BadMagic);
    }
    let (major, minor) = (bytes[6], bytes[7]);
    let (header_len, header_start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(NpyError::BufferTooShort {
                    need: 12,
                    have: bytes.len(),
                });
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12usize,
            )
        }
        _ => return Err(NpyError::UnsupportedVersion { major, minor }),
    };
    let header_end = header_start + header_len;
    if bytes.len() < header_end {
        return Err(NpyError::BufferTooShort {
            need: header_end,
            have: bytes.len(),
        });
    }
    let text = std::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|e| NpyError::BadHeader(e.to_string()))?;
    let dtype = DType::from_descr(&quoted_after(text, "'descr'")?)?;
    let fortran_order = match unquoted_after(text, "'fortran_order'")? {
        v if v.starts_with("True") => true,
        v if v.starts_with("False") => false,
        other => return Err(NpyError::BadHeader(format!("fortran_order {}", other))),
    };
    let shape = parse_shape(text)?;
    Ok(Header {
        dtype,
        fortran_order,
        shape,
        data_offset: header_end,
    })
}

/// Returns the first quoted string following `key`.
fn quoted_after(text: &str, key: &str) -> Result<String> {
    let start = text
        .find(key)
        .ok_or_else(|| NpyError::BadHeader(format!("{} missing", key)))?;
    let rest = &text[start + key.len()..];
    let open = rest
        .find('\'')
        .ok_or_else(|| NpyError::BadHeader(format!("{} value missing", key)))?;
    let after = &rest[open + 1..];
    let close = after
        .find('\'')
        .ok_or_else(|| NpyError::BadHeader(format!("{} value unterminated", key)))?;
    Ok(after[..close].to_string())
}

/// Returns the text following `key` up to the next comma (or closing brace).
fn unquoted_after(text: &str, key: &str) -> Result<String> {
    let start = text
        .find(key)
        .ok_or_else(|| NpyError::BadHeader(format!("{} missing", key)))?;
    let rest = &text[start + key.len()..];
    let rest = rest.trim_start_matches([':', ' ']);
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    Ok(rest[..end].trim().to_string())
}

/// Parses the `shape` tuple of the header dictionary.
///
/// The tuple is read as "from the first `(` to the next `)`, split on commas"
/// rather than by stopping at the next comma: a trailing comma or a comma
/// between dimensions would otherwise truncate `(281,)` to `(281`.
fn parse_shape(text: &str) -> Result<Vec<usize>> {
    let key = "'shape'";
    let start = text
        .find(key)
        .ok_or_else(|| NpyError::BadHeader(format!("{} missing", key)))?;
    let rest = &text[start + key.len()..];
    let open = rest
        .find('(')
        .ok_or_else(|| NpyError::BadHeader("shape tuple missing".into()))?;
    let close = rest[open..]
        .find(')')
        .ok_or_else(|| NpyError::BadHeader("shape tuple unterminated".into()))?;
    let inner = &rest[open + 1..open + close];
    let mut shape = Vec::new();
    for part in inner.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let value: usize = part
            .parse()
            .map_err(|_| NpyError::BadHeader(format!("shape element {}", part)))?;
        shape.push(value);
    }
    Ok(shape)
}

/// Reads a member of an `.npz` zip container.
///
/// The key may be given with or without the `.npy` suffix, e.g. `activations`
/// or `activations.npy`.
pub fn read_npz_member<P: AsRef<Path>>(path: P, key: &str) -> Result<NpyArray> {
    let path = path.as_ref().to_path_buf();
    let bytes = fs::read(&path).map_err(|source| NpyError::Io {
        path: path.clone(),
        source,
    })?;
    let leaf = if key.ends_with(".npy") {
        key.to_string()
    } else {
        format!("{}.npy", key)
    };
    let entries = zip_entries(&bytes)?;
    let entry = entries
        .into_iter()
        .find(|e| e.name == leaf || e.name == key)
        .ok_or_else(|| NpyError::MissingMember(key.to_string()))?;
    let data = zip_member_data(&bytes, &entry)?;
    parse_npy(data)
}

/// One entry of a zip central directory.
struct ZipEntry {
    name: String,
    method: u16,
    compressed_size: usize,
    local_header_offset: usize,
}

/// Lists the central directory entries of a zip container.
fn zip_entries(bytes: &[u8]) -> Result<Vec<ZipEntry>> {
    let eocd = find_eocd(bytes)?;
    let count = u16::from_le_bytes([bytes[eocd + 10], bytes[eocd + 11]]) as usize;
    let mut offset = u32::from_le_bytes([
        bytes[eocd + 16],
        bytes[eocd + 17],
        bytes[eocd + 18],
        bytes[eocd + 19],
    ]) as usize;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        if offset + 46 > bytes.len() || bytes[offset..offset + 4] != [0x50, 0x4b, 0x01, 0x02] {
            return Err(NpyError::BadHeader("bad zip central directory".into()));
        }
        let method = u16::from_le_bytes([bytes[offset + 10], bytes[offset + 11]]);
        let compressed_size = u32::from_le_bytes([
            bytes[offset + 20],
            bytes[offset + 21],
            bytes[offset + 22],
            bytes[offset + 23],
        ]) as usize;
        let name_len = u16::from_le_bytes([bytes[offset + 28], bytes[offset + 29]]) as usize;
        let extra_len = u16::from_le_bytes([bytes[offset + 30], bytes[offset + 31]]) as usize;
        let comment_len = u16::from_le_bytes([bytes[offset + 32], bytes[offset + 33]]) as usize;
        let local_header_offset = u32::from_le_bytes([
            bytes[offset + 42],
            bytes[offset + 43],
            bytes[offset + 44],
            bytes[offset + 45],
        ]) as usize;
        let name_start = offset + 46;
        let name_end = name_start + name_len;
        if name_end > bytes.len() {
            return Err(NpyError::BadHeader("truncated zip entry name".into()));
        }
        let name = String::from_utf8_lossy(&bytes[name_start..name_end]).to_string();
        entries.push(ZipEntry {
            name,
            method,
            compressed_size,
            local_header_offset,
        });
        offset = name_end + extra_len + comment_len;
    }
    Ok(entries)
}

/// Locates the end-of-central-directory record.
fn find_eocd(bytes: &[u8]) -> Result<usize> {
    if bytes.len() < 22 {
        return Err(NpyError::BadHeader(
            "file too small for a zip container".into(),
        ));
    }
    let sig = [0x50u8, 0x4b, 0x05, 0x06];
    let min = bytes.len().saturating_sub(66 * 1024);
    let mut found = None;
    let mut i = bytes.len().saturating_sub(22);
    loop {
        if bytes[i..i + 4] == sig {
            found = Some(i);
            break;
        }
        if i == min {
            break;
        }
        i -= 1;
    }
    found.ok_or_else(|| NpyError::BadHeader("zip end-of-central-directory not found".into()))
}

/// Extracts the raw bytes of a stored zip member.
fn zip_member_data<'a>(bytes: &'a [u8], entry: &ZipEntry) -> Result<&'a [u8]> {
    if entry.method != 0 {
        return Err(NpyError::UnsupportedZipMethod(entry.method));
    }
    let lho = entry.local_header_offset;
    if lho + 30 > bytes.len() || bytes[lho..lho + 4] != [0x50, 0x4b, 0x03, 0x04] {
        return Err(NpyError::BadHeader("bad zip local header".into()));
    }
    let name_len = u16::from_le_bytes([bytes[lho + 26], bytes[lho + 27]]) as usize;
    let extra_len = u16::from_le_bytes([bytes[lho + 28], bytes[lho + 29]]) as usize;
    let start = lho + 30 + name_len + extra_len;
    let end = start + entry.compressed_size;
    if end > bytes.len() {
        return Err(NpyError::BufferTooShort {
            need: end,
            have: bytes.len(),
        });
    }
    Ok(&bytes[start..end])
}

/// Directory holding the madmom parity fixtures.
///
/// Resolution order: `$MADMOM_FIXTURES`, then `tests/fixtures` next to the
/// crate, then the submodule path `third_party/madmom/tests/data`.
pub fn fixtures_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("MADMOM_FIXTURES") {
        return PathBuf::from(dir);
    }
    if let Some(manifest) = option_env!("CARGO_MANIFEST_DIR") {
        let base = Path::new(manifest);
        for candidate in ["tests/fixtures", "../../third_party/madmom/tests/data"] {
            let path = base.join(candidate);
            if path.is_dir() {
                return path;
            }
        }
    }
    PathBuf::from("tests/fixtures")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixtures() -> PathBuf {
        let dir = fixtures_dir();
        assert!(
            dir.is_dir(),
            "fixtures directory {} not found; set MADMOM_FIXTURES or init submodules",
            dir.display()
        );
        dir
    }

    /// Minimal in-memory `.npy` container, used where madmom has no stored fixture.
    fn npy_bytes(descr: &str, shape: &str, data: &[u8]) -> Vec<u8> {
        npy_bytes_ordered(descr, shape, data, false)
    }

    fn npy_bytes_ordered(descr: &str, shape: &str, data: &[u8], fortran: bool) -> Vec<u8> {
        let mut header = format!(
            "{{'descr': '{descr}', 'fortran_order': {}, 'shape': {shape}, }}",
            if fortran { "True" } else { "False" }
        );
        let padding = 64 - ((10 + header.len() + 1) % 64);
        header.push_str(&" ".repeat(if padding == 64 { 0 } else { padding }));
        header.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn reads_fortran_ordered_arrays_in_c_order() {
        // numpy stores a transposed view (madmom's processors end with `.T`) column-major;
        // the flag must be honoured or every consumer sees a transposed array
        let values: [f64; 6] = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // F order of [[1,2,3],[4,5,6]]
        let mut data = Vec::new();
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let bytes = npy_bytes_ordered("<f8", "(2, 3)", &data, true);
        let array = parse_npy(&bytes).expect("fortran ordered buffer");
        assert_eq!(array.shape(), &[2, 3]);
        assert!(!array.header.fortran_order, "the flag is resolved on read");
        let values = array.as_f32().expect("numeric");
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // a Fortran-ordered buffer must read like the same data in C order
        let mut c_data = Vec::new();
        for value in [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0] {
            c_data.extend_from_slice(&value.to_le_bytes());
        }
        let c_order = parse_npy(&npy_bytes("<f8", "(2, 3)", &c_data)).expect("c ordered buffer");
        assert_eq!(values, c_order.as_f32().expect("numeric"));
    }

    #[test]
    fn converts_integer_arrays_to_f32() {
        // int16 is how the reference dumps audio (`sample.signal.npy`, `frames.npy`);
        // without this conversion every framing comparison would be refused.
        let samples: [i16; 4] = [-32768, -1, 0, 32767];
        let mut data = Vec::new();
        for value in samples {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let bytes = npy_bytes("<i2", "(4,)", &data);
        let array = parse_npy(&bytes).unwrap();
        assert_eq!(array.dtype(), DType::I16);
        assert_eq!(array.as_f32().unwrap(), vec![-32768.0, -1.0, 0.0, 32767.0]);
    }

    #[test]
    fn reads_activation_member() {
        let path = fixtures().join("activations/sample.beats_blstm.npz");
        let arr = read_npz_member(&path, "activations").unwrap();
        assert_eq!(arr.dtype(), DType::F32);
        assert_eq!(arr.shape(), &[281]);
        let values = arr.as_f32().unwrap();
        assert_eq!(values.len(), 281);
        let expected = [
            0.004_785_209_f32,
            0.006_030_515_f32,
            0.009_267_444_f32,
            0.014_188_592_f32,
            0.023_421_122_f32,
        ];
        for (got, want) in values.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "first values differ: {} vs {}",
                got,
                want
            );
        }
        let sum: f32 = values.iter().sum();
        assert!((sum - 9.354_218).abs() < 1e-3, "sum differs: {}", sum);
    }

    #[test]
    fn reads_zero_dimensional_fps() {
        let path = fixtures().join("activations/sample.beats_blstm.npz");
        let fps = read_npz_member(&path, "fps").unwrap();
        assert_eq!(fps.dtype(), DType::I64);
        assert!(fps.shape().is_empty());
        assert_eq!(fps.scalar_f64().unwrap(), 100.0);

        let dc = read_npz_member(
            fixtures().join("activations/sample.deep_chroma.npz"),
            "fps.npy",
        )
        .unwrap();
        assert_eq!(dc.scalar_f64().unwrap(), 10.0);
    }

    #[test]
    fn reads_two_dimensional_member() {
        let path = fixtures().join("activations/sample.deep_chroma.npz");
        let arr = read_npz_member(&path, "activations").unwrap();
        assert_eq!(arr.shape(), &[29, 12]);
        assert_eq!(arr.data.len(), 29 * 12 * 4);
        let values = arr.as_f32().unwrap();
        assert!((values[0] - 0.104_024_11).abs() < 1e-6);
        assert!((values[1] - 0.047_728_71).abs() < 1e-6);
        assert!((values[2] - 0.007_253_505).abs() < 1e-6);
    }

    #[test]
    fn object_arrays_are_reported_not_guessed() {
        let path = fixtures().join("activations/sample.beats_tcn_tempo.npz");
        let fps = read_npz_member(&path, "fps").unwrap();
        assert_eq!(fps.dtype(), DType::Object);
        assert!(matches!(fps.scalar_f64(), Err(NpyError::ObjectArray)));
        // the activations of the same file are still readable
        let act = read_npz_member(&path, "activations").unwrap();
        assert_eq!(act.shape(), &[300]);
    }

    #[test]
    fn missing_member_is_reported() {
        let path = fixtures().join("activations/sample.beats_blstm.npz");
        match read_npz_member(&path, "nope") {
            Err(NpyError::MissingMember(name)) => assert_eq!(name, "nope"),
            other => panic!("expected MissingMember, got {:?}", other),
        }
    }

    #[test]
    fn header_is_padded_and_little_endian() {
        let path = fixtures().join("activations/sample.beats_blstm.npz");
        let raw = read_npz_member(&path, "activations").unwrap();
        let bytes = fs::read(&path).unwrap();
        let entries = zip_entries(&bytes).unwrap();
        assert!(entries.iter().any(|e| e.name == "activations.npy"));
        let member = header_range(&bytes, "activations.npy");
        let header = parse_header(member).unwrap();
        // NumPy pads the header text with spaces and terminates it with a
        // newline; the data starts right after it.
        assert_eq!(member[header.data_offset - 1], b'\n');
        assert!(header.data_offset > 10);
        assert!(!header.fortran_order);
        assert_eq!(raw.header, header);
    }

    /// Helper: slices out the stored member bytes so `parse_header` can be
    /// called on them directly.
    fn header_range<'a>(bytes: &'a [u8], name: &str) -> &'a [u8] {
        let entry = zip_entries(bytes)
            .unwrap()
            .into_iter()
            .find(|e| e.name == name)
            .unwrap();
        zip_member_data(bytes, &entry).unwrap()
    }
}
