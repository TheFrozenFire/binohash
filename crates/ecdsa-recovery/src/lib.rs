use std::cmp::Ordering;

use der::ParsedDerSig;
pub use secp256k1::PublicKey;
use secp256k1::{
    Message, Secp256k1,
    ecdsa::{RecoverableSignature, RecoveryId, Signature},
};

/// secp256k1 curve order N.
pub const SECP256K1_N: [u8; 32] = [
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xfe, 0xba, 0xae, 0xdc, 0xe6, 0xaf, 0x48, 0xa0, 0x3b, 0xbf, 0xd2, 0x5e, 0x8c, 0xd0, 0x36,
    0x41, 0x41,
];

#[derive(Debug, thiserror::Error)]
pub enum RecoveryError {
    #[error("invalid recovery id: {0}")]
    InvalidRecoveryId(u8),
    #[error("invalid compact signature: {0}")]
    InvalidSignature(secp256k1::Error),
    #[error("key recovery failed: {0}")]
    RecoveryFailed(secp256k1::Error),
    #[error("recovered key does not verify: {0}")]
    VerificationFailed(secp256k1::Error),
    #[error("no valid recovery id found")]
    NoValidRecoveryId,
}

/// Recover a public key from a parsed DER signature and message digest.
///
/// The `recid` selects which candidate point R to use (0 or 1, corresponding
/// to the two possible y-coordinates for a given x on secp256k1).
pub fn recover_pubkey(
    parsed: &ParsedDerSig,
    digest: [u8; 32],
    recid: u8,
) -> Result<PublicKey, RecoveryError> {
    let secp = Secp256k1::verification_only();
    let compact = to_compact(parsed);
    let recovery_id =
        RecoveryId::try_from(recid as i32).map_err(|_| RecoveryError::InvalidRecoveryId(recid))?;
    let rec_sig = RecoverableSignature::from_compact(&compact, recovery_id)
        .map_err(RecoveryError::InvalidSignature)?;
    let msg = Message::from_digest(digest);
    let pubkey = secp
        .recover_ecdsa(msg, &rec_sig)
        .map_err(RecoveryError::RecoveryFailed)?;

    // Verify the recovered key actually satisfies the signature
    let std_sig: Signature = rec_sig.to_standard();
    secp.verify_ecdsa(msg, &std_sig, &pubkey)
        .map_err(RecoveryError::VerificationFailed)?;

    Ok(pubkey)
}

/// Recover a public key trying recovery IDs 0 and 1, returning the first success.
pub fn recover_first_pubkey(
    parsed: &ParsedDerSig,
    digest: [u8; 32],
) -> Result<(PublicKey, u8), RecoveryError> {
    for recid in 0..=1u8 {
        match recover_pubkey(parsed, digest, recid) {
            Ok(pubkey) => return Ok((pubkey, recid)),
            Err(_) => continue,
        }
    }
    Err(RecoveryError::NoValidRecoveryId)
}

/// Convert a `ParsedDerSig` into the 64-byte compact (r || s) format used by libsecp256k1.
fn to_compact(parsed: &ParsedDerSig) -> [u8; 64] {
    let mut compact = [0u8; 64];
    compact[..32].copy_from_slice(&parsed.r);
    compact[32..].copy_from_slice(&parsed.s);
    compact
}

/// Check if a 32-byte value is a valid secp256k1 scalar (non-zero and less than N).
pub fn is_valid_scalar(bytes: &[u8; 32]) -> bool {
    if bytes.iter().all(|b| *b == 0) {
        return false;
    }
    compare_be(bytes, &SECP256K1_N) == Ordering::Less
}

/// Compare two big-endian byte arrays lexicographically.
fn compare_be<const N: usize>(left: &[u8; N], right: &[u8; N]) -> Ordering {
    left.iter()
        .zip(right.iter())
        .find_map(|(l, r)| match l.cmp(r) {
            Ordering::Equal => None,
            ord => Some(ord),
        })
        .unwrap_or(Ordering::Equal)
}

/// Deterministically derive a valid secp256k1 x-coordinate from a label string.
///
/// Iterates `SHA256(label || counter)` until finding a value that is both a valid
/// scalar and a valid x-coordinate on the curve. Used for constructing hardcoded
/// nonce signatures.
pub fn derive_valid_xcoord(label: &str) -> [u8; 32] {
    for counter in 0u32..u32::MAX {
        let mut seed = Vec::with_capacity(label.len() + 4);
        seed.extend_from_slice(label.as_bytes());
        seed.extend_from_slice(&counter.to_be_bytes());
        let candidate = hash::sha256(&seed);
        if !is_valid_scalar(&candidate) {
            continue;
        }
        let mut compressed = [0u8; 33];
        compressed[0] = 0x02;
        compressed[1..].copy_from_slice(&candidate);
        if PublicKey::from_slice(&compressed).is_ok() {
            return candidate;
        }
    }
    unreachable!("failed to derive valid secp256k1 x-coordinate")
}

/// Deterministically derive a valid secp256k1 scalar from a label string.
///
/// Iterates `SHA256(label || counter)` until finding a non-zero value less than N.
/// Used for constructing the s component of hardcoded nonce signatures.
pub fn derive_valid_scalar(label: &str) -> [u8; 32] {
    for counter in 0u32..u32::MAX {
        let mut seed = Vec::with_capacity(label.len() + 4);
        seed.extend_from_slice(label.as_bytes());
        seed.extend_from_slice(&counter.to_be_bytes());
        let candidate = hash::sha256(&seed);
        if is_valid_scalar(&candidate) {
            return candidate;
        }
    }
    unreachable!("failed to derive valid scalar")
}

/// Find all values in [1, 127] that are valid x-coordinates on secp256k1.
///
/// These small r values are used for constructing 9-byte minimum DER dummy
/// signatures (where r fits in a single byte without padding).
pub fn small_r_values() -> Vec<u8> {
    (1u16..128)
        .filter_map(|r| {
            let mut x = [0u8; 32];
            x[31] = r as u8;
            let mut compressed = [0u8; 33];
            compressed[0] = 0x02;
            compressed[1..].copy_from_slice(&x);
            PublicKey::from_slice(&compressed).ok()?;
            Some(r as u8)
        })
        .collect()
}
