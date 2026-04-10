use der::ParsedDerSig;
pub use ecdsa_recovery::PublicKey;

/// Controls whether the puzzle uses strict BIP66 DER validation or a relaxed
/// predicate for fast integration testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Requires `is_valid_der_sig` — the real ~2^-46 predicate.
    Production,
    /// Accepts either strict DER or the easy predicate (first nibble == 0x3),
    /// giving ~1/16 hit rate for fast testing.
    EasyTest,
}

/// A successful puzzle hit: the HASH160 of the recovered key is a valid signature.
#[derive(Debug, Clone)]
pub struct PuzzleHit {
    /// The public key recovered from (sig_nonce, sighash).
    pub key_nonce: PublicKey,
    /// HASH160(key_nonce) — the 20-byte value that satisfies the DER predicate.
    pub sig_puzzle: [u8; 20],
    /// Whether sig_puzzle passes strict BIP66 DER validation (vs. easy predicate only).
    pub is_strict_der: bool,
    /// Which recovery ID (0 or 1) produced key_nonce.
    pub recovery_id: u8,
}

/// Attempt to recover a public key from a nonce signature and message digest.
///
/// Tries recovery IDs 0 and 1, returning the first success. Returns `None`
/// if recovery fails for both IDs.
pub fn try_recover_key_nonce(
    sig_nonce: &ParsedDerSig,
    digest: [u8; 32],
) -> Option<(PublicKey, u8)> {
    ecdsa_recovery::recover_first_pubkey(sig_nonce, digest).ok()
}

/// Hash a recovered public key and check if the output is valid DER.
///
/// Returns the 20-byte HASH160 output and whether it passes strict DER validation.
pub fn check_hash_to_sig(key_nonce: &PublicKey) -> ([u8; 20], bool) {
    let sig_puzzle = hash::hash160(&key_nonce.serialize());
    let is_der = der::is_valid_der_sig(&sig_puzzle);
    (sig_puzzle, is_der)
}

/// Evaluate the hash-to-sig puzzle for a given nonce signature and digest.
///
/// This is the core QSB predicate:
/// 1. Recover `key_nonce = Recover(sig_nonce, digest)`
/// 2. Compute `sig_puzzle = HASH160(key_nonce)`
/// 3. Check if `sig_puzzle` satisfies the DER predicate (strict or easy depending on mode)
///
/// Returns `Some(PuzzleHit)` if the predicate passes, `None` otherwise.
pub fn evaluate_puzzle(
    sig_nonce: &ParsedDerSig,
    digest: [u8; 32],
    mode: SearchMode,
) -> Option<PuzzleHit> {
    let (key_nonce, recovery_id) = try_recover_key_nonce(sig_nonce, digest)?;
    let (sig_puzzle, is_strict_der) = check_hash_to_sig(&key_nonce);

    let passes = match mode {
        SearchMode::Production => is_strict_der,
        SearchMode::EasyTest => is_strict_der || der::easy_der_predicate(&sig_puzzle),
    };

    if !passes {
        return None;
    }

    Some(PuzzleHit {
        key_nonce,
        sig_puzzle,
        is_strict_der,
        recovery_id,
    })
}
