/// A parsed DER-encoded ECDSA signature with r and s as 32-byte big-endian arrays.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedDerSig {
    pub r: [u8; 32],
    pub s: [u8; 32],
    pub sighash_type: u8,
}

/// Check if raw bytes constitute a valid BIP66 strict DER-encoded ECDSA signature.
///
/// Accepts variable-length input (works for both 20-byte RIPEMD-160 and 32-byte SHA-256
/// outputs). The trailing byte is treated as the sighash type but is not validated —
/// SCRIPT_VERIFY_STRICTENC is relay policy only, not consensus.
pub fn is_valid_der_sig(data: &[u8]) -> bool {
    // Minimum: 30 06 02 01 xx 02 01 xx sh = 9 bytes
    if data.len() < 9 || data[0] != 0x30 {
        return false;
    }

    let total_len = data[1] as usize;
    if total_len + 3 != data.len() {
        return false;
    }

    let mut idx = 2usize;
    for _ in 0..2 {
        if idx >= data.len().saturating_sub(1) || data[idx] != 0x02 {
            return false;
        }
        idx += 1;
        if idx >= data.len().saturating_sub(1) {
            return false;
        }
        let int_len = data[idx] as usize;
        idx += 1;
        if int_len == 0 || idx + int_len > data.len().saturating_sub(1) {
            return false;
        }
        // Negative: MSB set
        if data[idx] & 0x80 != 0 {
            return false;
        }
        // Unnecessary leading zero: 0x00 followed by byte with MSB clear
        if int_len > 1 && data[idx] == 0x00 && data[idx + 1] & 0x80 == 0 {
            return false;
        }
        idx += int_len;
    }

    // Should have consumed exactly up to the sighash byte
    idx == data.len().saturating_sub(1)
}

/// Parse a DER-encoded signature into its r, s components and sighash type.
///
/// Returns `None` if the signature is not valid DER.
pub fn parse_der_sig(data: &[u8]) -> Option<ParsedDerSig> {
    if !is_valid_der_sig(data) {
        return None;
    }
    let sighash_type = *data.last()?;
    let mut idx = 2usize;

    let read_int = |bytes: &[u8], idx: &mut usize| -> Option<[u8; 32]> {
        if *idx + 2 > bytes.len() {
            return None;
        }
        if bytes[*idx] != 0x02 {
            return None;
        }
        *idx += 1;
        let int_len = bytes[*idx] as usize;
        *idx += 1;
        if *idx + int_len > bytes.len().saturating_sub(1) {
            return None;
        }
        let int_bytes = &bytes[*idx..*idx + int_len];
        *idx += int_len;
        // Strip leading zero padding
        let no_prefix = if int_bytes.first() == Some(&0) {
            &int_bytes[1..]
        } else {
            int_bytes
        };
        if no_prefix.len() > 32 {
            return None;
        }
        let mut out = [0u8; 32];
        out[32 - no_prefix.len()..].copy_from_slice(no_prefix);
        Some(out)
    };

    let r = read_int(data, &mut idx)?;
    let s = read_int(data, &mut idx)?;
    if idx != data.len().saturating_sub(1) {
        return None;
    }
    Some(ParsedDerSig { r, s, sighash_type })
}

/// Trim leading zeros from a byte slice, preserving at least one byte.
fn trim_leading_zeros(bytes: &[u8]) -> &[u8] {
    let idx = bytes
        .iter()
        .position(|b| *b != 0)
        .unwrap_or(bytes.len().saturating_sub(1));
    &bytes[idx..]
}

/// Encode a 32-byte big-endian integer as DER integer bytes (without the 0x02 tag/length).
///
/// Trims leading zeros and adds a 0x00 padding byte if the MSB would be set (indicating
/// a negative value in DER).
pub fn encode_der_integer(value: &[u8; 32]) -> Vec<u8> {
    let trimmed = trim_leading_zeros(value);
    let mut out = if trimmed.is_empty() {
        vec![0]
    } else {
        trimmed.to_vec()
    };
    if out[0] & 0x80 != 0 {
        out.insert(0, 0);
    }
    out
}

/// Encode r and s values as a DER-encoded ECDSA signature with a trailing sighash byte.
///
/// Output format: `30 [inner_len] 02 [r_len] [r] 02 [s_len] [s] [sighash_type]`
pub fn encode_der_sig(r: &[u8; 32], s: &[u8; 32], sighash_type: u8) -> Vec<u8> {
    let r_bytes = encode_der_integer(r);
    let s_bytes = encode_der_integer(s);

    let inner_len = 2 + r_bytes.len() + 2 + s_bytes.len();
    let mut out = Vec::with_capacity(inner_len + 3);
    out.push(0x30);
    out.push(inner_len as u8);
    out.push(0x02);
    out.push(r_bytes.len() as u8);
    out.extend_from_slice(&r_bytes);
    out.push(0x02);
    out.push(s_bytes.len() as u8);
    out.extend_from_slice(&s_bytes);
    out.push(sighash_type);
    out
}

/// Fast predicate: checks if the first nibble of the input is 0x3.
///
/// Used as a relaxed filter in easy-test mode — much higher hit rate than strict DER
/// validation (~1/16 vs ~2^-46), enabling fast integration testing.
pub fn easy_der_predicate(data: &[u8]) -> bool {
    data.first().is_some_and(|b| b >> 4 == 0x3)
}
