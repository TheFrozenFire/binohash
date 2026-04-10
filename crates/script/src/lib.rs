// Bitcoin Script opcodes
pub const OP_0: u8 = 0x00;
pub const OP_PUSHDATA1: u8 = 0x4c;
pub const OP_PUSHDATA2: u8 = 0x4d;
pub const OP_1: u8 = 0x51;
pub const OP_2: u8 = 0x52;
pub const OP_16: u8 = 0x60;
pub const OP_DUP: u8 = 0x76;
pub const OP_OVER: u8 = 0x78;
pub const OP_ROLL: u8 = 0x7a;
pub const OP_SWAP: u8 = 0x7c;
pub const OP_EQUALVERIFY: u8 = 0x88;
pub const OP_ADD: u8 = 0x93;
pub const OP_MIN: u8 = 0xa3;
pub const OP_RIPEMD160: u8 = 0xa6;
pub const OP_HASH160: u8 = 0xa9;
pub const OP_CHECKSIG: u8 = 0xac;
pub const OP_CHECKSIGVERIFY: u8 = 0xad;
pub const OP_CHECKMULTISIG: u8 = 0xae;

/// Encode data as a Bitcoin Script push operation.
///
/// - Empty data → OP_0
/// - 1-75 bytes → single-byte length prefix
/// - 76-255 bytes → OP_PUSHDATA1 + 1-byte length
/// - 256-65535 bytes → OP_PUSHDATA2 + 2-byte little-endian length
pub fn push_data(data: &[u8]) -> Vec<u8> {
    match data.len() {
        0 => vec![OP_0],
        n @ 1..=75 => {
            let mut out = Vec::with_capacity(n + 1);
            out.push(n as u8);
            out.extend_from_slice(data);
            out
        }
        n @ 76..=255 => {
            let mut out = Vec::with_capacity(n + 2);
            out.push(OP_PUSHDATA1);
            out.push(n as u8);
            out.extend_from_slice(data);
            out
        }
        n @ 256..=65535 => {
            let mut out = Vec::with_capacity(n + 3);
            out.push(OP_PUSHDATA2);
            out.extend_from_slice(&(n as u16).to_le_bytes());
            out.extend_from_slice(data);
            out
        }
        n => panic!("push_data: data too large ({n} bytes)"),
    }
}

/// Encode an integer as a Bitcoin Script number push.
///
/// - 0 → OP_0
/// - 1-16 → OP_1 through OP_16
/// - Other → minimal little-endian encoding with sign bit padding
pub fn push_number(value: i64) -> Vec<u8> {
    match value {
        0 => vec![OP_0],
        1..=16 => vec![OP_1 + value as u8 - 1],
        _ => {
            let mut n = value as u64;
            let mut encoded = Vec::new();
            while n > 0 {
                encoded.push((n & 0xff) as u8);
                n >>= 8;
            }
            if encoded.last().is_some_and(|b| b & 0x80 != 0) {
                encoded.push(0);
            }
            push_data(&encoded)
        }
    }
}

/// Remove all occurrences of `push_data(sig_data)` from a script.
///
/// This implements the FindAndDelete operation applied before legacy sighash
/// computation. The pattern is the sig data wrapped in its push encoding.
pub fn find_and_delete(script: &[u8], sig_data: &[u8]) -> Vec<u8> {
    let pattern = push_data(sig_data);
    let mut result = Vec::with_capacity(script.len());
    let mut i = 0usize;
    while i + pattern.len() <= script.len() {
        if script[i..i + pattern.len()] == pattern {
            i += pattern.len();
        } else {
            result.push(script[i]);
            i += 1;
        }
    }
    result.extend_from_slice(&script[i..]);
    result
}

/// Count the number of non-push opcodes in a script.
///
/// Opcodes above OP_16 (0x60) are counted. Push opcodes (0x01-0x4b, OP_PUSHDATA1,
/// OP_PUSHDATA2) and number opcodes (OP_0, OP_1-OP_16) are not counted.
pub fn count_non_push_opcodes(script: &[u8]) -> Result<usize, ScriptError> {
    let mut i = 0usize;
    let mut count = 0usize;
    while i < script.len() {
        let opcode = script[i];
        i += 1;
        match opcode {
            0x01..=0x4b => i += opcode as usize,
            OP_PUSHDATA1 => {
                if i >= script.len() {
                    return Err(ScriptError::Truncated("OP_PUSHDATA1"));
                }
                let len = script[i] as usize;
                i += 1 + len;
            }
            OP_PUSHDATA2 => {
                if i + 1 >= script.len() {
                    return Err(ScriptError::Truncated("OP_PUSHDATA2"));
                }
                let len = u16::from_le_bytes([script[i], script[i + 1]]) as usize;
                i += 2 + len;
            }
            _ => {
                if opcode > OP_16 {
                    count += 1;
                }
            }
        }
        if i > script.len() {
            return Err(ScriptError::PushOverrun);
        }
    }
    Ok(count)
}

#[derive(Debug, thiserror::Error)]
pub enum ScriptError {
    #[error("truncated {0}")]
    Truncated(&'static str),
    #[error("push data overruns script length")]
    PushOverrun,
}
