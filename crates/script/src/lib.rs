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
    #[error("script exceeds {limit} byte size limit ({actual} bytes)")]
    SizeExceeded { limit: usize, actual: usize },
    #[error("script exceeds {limit} non-push opcode limit ({actual} opcodes)")]
    OpcodeExceeded { limit: usize, actual: usize },
}

/// QSB script configuration parameters.
#[derive(Debug, Clone, Copy)]
pub struct QsbConfig {
    /// Pool size (number of dummy sigs / HORS entries per round).
    pub n: usize,
    /// Number of signed selections in round 1.
    pub t1_signed: usize,
    /// Number of bonus selections in round 1.
    pub t1_bonus: usize,
    /// Number of signed selections in round 2.
    pub t2_signed: usize,
    /// Number of bonus selections in round 2.
    pub t2_bonus: usize,
}

impl QsbConfig {
    pub fn t1_total(self) -> usize {
        self.t1_signed + self.t1_bonus
    }

    pub fn t2_total(self) -> usize {
        self.t2_signed + self.t2_bonus
    }

    /// Config A from the paper: t=8+1b, 7+2b, n=150. Fits exactly 201 opcodes.
    pub fn config_a() -> Self {
        Self { n: 150, t1_signed: 8, t1_bonus: 1, t2_signed: 7, t2_bonus: 2 }
    }

    /// Small test config for integration testing.
    pub fn test() -> Self {
        Self { n: 20, t1_signed: 2, t1_bonus: 0, t2_signed: 2, t2_bonus: 0 }
    }
}

/// Build the 5-opcode pinning section of the locking script.
///
/// Witness expects: `<key_puzzle> <key_nonce>` (key_nonce on top)
pub fn build_pinning_script(sig_nonce: &[u8]) -> Vec<u8> {
    let mut s = Vec::new();
    s.extend_from_slice(&push_data(sig_nonce));
    s.push(OP_OVER);
    s.push(OP_CHECKSIGVERIFY);
    s.push(OP_RIPEMD160);
    s.push(OP_SWAP);
    s.push(OP_CHECKSIGVERIFY);
    s
}

/// Build a single round's script section.
///
/// Includes: HORS commitments, dummy sigs, OP_0, sig_nonce, selection logic,
/// puzzle derivation, CHECKSIGVERIFY, and CHECKMULTISIG.
pub fn build_round_script(
    n: usize,
    t_signed: usize,
    t_bonus: usize,
    sig_nonce: &[u8],
    hors_commitments: &[[u8; 20]],
    dummy_sigs: &[[u8; 9]],
) -> Vec<u8> {
    let t_total = t_signed + t_bonus;
    let mut s = Vec::new();

    // Push n HORS commitments (reversed order)
    for commitment in hors_commitments.iter().rev() {
        s.extend_from_slice(&push_data(commitment));
    }
    // Push n dummy sigs (reversed order)
    for dummy in dummy_sigs.iter().rev() {
        s.extend_from_slice(&push_data(dummy));
    }
    // OP_0 (CHECKMULTISIG dummy) + hardcoded sig_nonce
    s.push(OP_0);
    s.extend_from_slice(&push_data(sig_nonce));

    // Signed selections (9 ops each)
    for i in 0..t_signed {
        let idx_pos = 2 * n + 1 - i;
        let sanitize = n - i;
        let preimage_pos = 2 * n + 1 + t_total - 2 * i;
        s.extend_from_slice(&push_number(idx_pos as i64));
        s.push(OP_ROLL);
        s.extend_from_slice(&push_number(sanitize as i64));
        s.push(OP_MIN);
        s.push(OP_DUP);
        s.extend_from_slice(&push_number((n + 1) as i64));
        s.push(OP_ADD);
        s.push(OP_ROLL);
        s.extend_from_slice(&push_number(preimage_pos as i64));
        s.push(OP_ROLL);
        s.push(OP_HASH160);
        s.push(OP_EQUALVERIFY);
        s.push(OP_ROLL);
    }

    // Bonus selections (3 ops each)
    for i in 0..t_bonus {
        let j = t_signed + i;
        let idx_pos = 2 * n + 1 - j;
        let sanitize = n - j;
        s.extend_from_slice(&push_number(idx_pos as i64));
        s.push(OP_ROLL);
        s.extend_from_slice(&push_number(sanitize as i64));
        s.push(OP_MIN);
        s.push(OP_ROLL);
    }

    // Puzzle: ROLL key_nonce + DUP + RIPEMD160
    let puzzle_pos = 2 * n + 2;
    s.extend_from_slice(&push_number(puzzle_pos as i64));
    s.push(OP_ROLL);
    s.push(OP_DUP);
    s.push(OP_RIPEMD160);

    // CHECKSIGVERIFY for sig_puzzle
    s.extend_from_slice(&push_number(puzzle_pos as i64));
    s.push(OP_ROLL);
    s.push(OP_CHECKSIGVERIFY);

    // CHECKMULTISIG (t+1)-of-(t+1)
    let m = t_total + 1;
    s.extend_from_slice(&push_number(m as i64));
    s.push(OP_2);
    s.push(OP_ROLL);

    let cms_roll_pos = 2 * n + 3;
    for _ in 0..t_total {
        s.extend_from_slice(&push_number(cms_roll_pos as i64));
        s.push(OP_ROLL);
    }

    s.extend_from_slice(&push_number(m as i64));
    s.push(OP_CHECKMULTISIG);
    s
}

/// Build the complete QSB locking script (pinning + 2 rounds).
pub fn build_full_script(
    config: QsbConfig,
    pin_sig: &[u8],
    round1_sig: &[u8],
    round2_sig: &[u8],
    hors_commitments: &[Vec<[u8; 20]>; 2],
    dummy_sigs: &[Vec<[u8; 9]>; 2],
) -> Vec<u8> {
    let mut s = build_pinning_script(pin_sig);
    s.extend_from_slice(&build_round_script(
        config.n, config.t1_signed, config.t1_bonus,
        round1_sig, &hors_commitments[0], &dummy_sigs[0],
    ));
    s.extend_from_slice(&build_round_script(
        config.n, config.t2_signed, config.t2_bonus,
        round2_sig, &hors_commitments[1], &dummy_sigs[1],
    ));
    s
}

/// Validate that a script fits within Bitcoin consensus limits.
pub fn validate_script_limits(script: &[u8]) -> Result<(), ScriptError> {
    const MAX_SCRIPT_SIZE: usize = 10_000;
    const MAX_NON_PUSH_OPCODES: usize = 201;

    if script.len() > MAX_SCRIPT_SIZE {
        return Err(ScriptError::SizeExceeded {
            limit: MAX_SCRIPT_SIZE,
            actual: script.len(),
        });
    }
    let opcode_count = count_non_push_opcodes(script)?;
    if opcode_count > MAX_NON_PUSH_OPCODES {
        return Err(ScriptError::OpcodeExceeded {
            limit: MAX_NON_PUSH_OPCODES,
            actual: opcode_count,
        });
    }
    Ok(())
}
