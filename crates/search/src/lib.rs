use der::ParsedDerSig;
use puzzle::{PuzzleHit, SearchMode, evaluate_puzzle};
use rayon::prelude::*;
use script::find_and_delete;
use subset::CombinationIter;
use tx::Transaction;

/// Parameters for the pinning puzzle search.
pub struct PinningSearchParams<'a> {
    /// The template transaction (sequence and locktime will be varied).
    pub tx: &'a Transaction,
    /// The full locking script (before any FindAndDelete).
    pub full_script: &'a [u8],
    /// The script code after removing the pinning sig via FindAndDelete.
    pub pin_script_code: &'a [u8],
    /// The parsed pinning nonce signature.
    pub sig_nonce: &'a ParsedDerSig,
    /// The raw DER-encoded nonce signature bytes (for FindAndDelete).
    pub sig_nonce_bytes: &'a [u8],
    /// The search space for (sequence, locktime) pairs.
    pub search_space: PinningSearchSpace,
    /// Whether to use strict DER or the easy-test predicate.
    pub mode: SearchMode,
    /// Which input index to compute the sighash for.
    pub input_index: usize,
}

/// Defines the (sequence, locktime) search space for pinning.
#[derive(Clone)]
pub struct PinningSearchSpace {
    pub sequence_start: u32,
    pub sequence_count: u32,
    pub locktime_start: u32,
    pub locktime_count: u32,
}

/// A successful pinning search result.
pub struct PinningHit {
    pub sequence: u32,
    pub locktime: u32,
    pub puzzle_hit: PuzzleHit,
}

/// Parameters for the digest (round) search.
pub struct DigestSearchParams<'a> {
    /// The transaction with pinning parameters already fixed.
    pub tx: &'a Transaction,
    /// The full locking script.
    pub full_script: &'a [u8],
    /// The parsed round nonce signature.
    pub sig_nonce: &'a ParsedDerSig,
    /// The raw DER-encoded round nonce signature bytes (for FindAndDelete).
    pub sig_nonce_bytes: &'a [u8],
    /// The dummy signatures for this round.
    pub dummy_sigs: &'a [[u8; 9]],
    /// The HORS secrets for this round (revealed for signed indices).
    pub hors_secrets: &'a [[u8; 20]],
    /// Pool size.
    pub n: usize,
    /// Number of signed selections per round.
    pub t_signed: usize,
    /// Number of bonus selections per round.
    pub t_bonus: usize,
    /// Strict DER or easy-test mode.
    pub mode: SearchMode,
    /// Which input index to compute the sighash for.
    pub input_index: usize,
}

/// A successful digest search result.
pub struct DigestHit {
    /// The signed subset indices (first t_signed elements participated in HORS).
    pub signed_indices: Vec<usize>,
    /// The bonus subset indices (participate in FindAndDelete but skip HORS).
    pub bonus_indices: Vec<usize>,
    /// All selected indices (signed ++ bonus), for convenience.
    pub indices: Vec<usize>,
    /// The puzzle hit for this subset.
    pub puzzle_hit: PuzzleHit,
    /// The recovered key_puzzle (if sig_puzzle was strict DER).
    pub key_puzzle: Option<ecdsa_recovery::PublicKey>,
}

/// Search for a pinning solution by iterating over (sequence, locktime) pairs.
///
/// Uses rayon to parallelize across the search space. Returns the first hit
/// found by any worker, or `None` if exhausted.
pub fn search_pinning(params: PinningSearchParams<'_>) -> Option<PinningHit> {
    let ss = &params.search_space;
    let total = ss.sequence_count as u64 * ss.locktime_count as u64;
    let lt_count = ss.locktime_count as u64;

    (0..total).into_par_iter().find_map_first(|offset| {
        let sequence = ss.sequence_start.wrapping_add((offset / lt_count) as u32);
        let locktime = ss.locktime_start.wrapping_add((offset % lt_count) as u32);

        let mut tx = params.tx.clone();
        tx.inputs[params.input_index].sequence = sequence;
        tx.locktime = locktime;

        let digest = tx
            .legacy_sighash(
                params.input_index,
                params.pin_script_code,
                params.sig_nonce.sighash_type,
            )
            .ok()?;

        let hit = evaluate_puzzle(params.sig_nonce, digest, params.mode)?;
        Some(PinningHit {
            sequence,
            locktime,
            puzzle_hit: hit,
        })
    })
}

/// Search for a digest solution by iterating over all C(n, t_total) subsets.
///
/// For each subset, applies FindAndDelete to remove the selected dummy signatures
/// and the round nonce signature from the full script, then computes the sighash
/// and evaluates the puzzle.
///
/// Uses rayon to parallelize by collecting subsets into chunks. Returns the first
/// hit, or `None`.
pub fn search_digest(params: DigestSearchParams<'_>) -> Option<DigestHit> {
    let t_total = params.t_signed + params.t_bonus;

    // Pre-compute the script with the round nonce sig removed
    let base_script_code = find_and_delete(params.full_script, params.sig_nonce_bytes);

    // Collect subsets into chunks for parallel evaluation.
    // For small n (test configs), this is fast. For production n=150 t=9,
    // the iterator itself is the bottleneck and would need a different approach
    // (e.g., index-based parallel ranges with nth_combination).
    let chunk_size = 1024;
    let mut combos: Vec<Vec<usize>> = Vec::with_capacity(chunk_size);

    for combo in CombinationIter::new(params.n, t_total) {
        combos.push(combo);
        if combos.len() >= chunk_size {
            if let Some(hit) = evaluate_chunk(&combos, &params, &base_script_code) {
                return Some(hit);
            }
            combos.clear();
        }
    }
    // Process remaining combos
    if !combos.is_empty() {
        if let Some(hit) = evaluate_chunk(&combos, &params, &base_script_code) {
            return Some(hit);
        }
    }

    None
}

fn evaluate_chunk(
    combos: &[Vec<usize>],
    params: &DigestSearchParams<'_>,
    base_script_code: &[u8],
) -> Option<DigestHit> {
    combos.par_iter().find_map_first(|combo| {
        let mut script_code = base_script_code.to_vec();
        for &index in combo {
            script_code = find_and_delete(&script_code, &params.dummy_sigs[index]);
        }

        let digest = params
            .tx
            .legacy_sighash(
                params.input_index,
                &script_code,
                params.sig_nonce.sighash_type,
            )
            .ok()?;

        let hit = evaluate_puzzle(params.sig_nonce, digest, params.mode)?;

        // Recover key_puzzle if sig_puzzle is strict DER
        let key_puzzle = if hit.is_strict_der {
            recover_key_puzzle(&hit, params.tx, params.full_script, params.input_index)
        } else {
            None
        };

        let signed_indices = combo[..params.t_signed].to_vec();
        let bonus_indices = combo[params.t_signed..].to_vec();

        Some(DigestHit {
            signed_indices,
            bonus_indices,
            indices: combo.clone(),
            puzzle_hit: hit,
            key_puzzle,
        })
    })
}

/// Recover key_puzzle from a puzzle hit.
///
/// When sig_puzzle is valid DER, it acts as a signature. We parse it, compute the
/// sighash against the script with sig_puzzle removed (via FindAndDelete), and
/// recover the corresponding public key.
fn recover_key_puzzle(
    hit: &PuzzleHit,
    tx: &Transaction,
    full_script: &[u8],
    input_index: usize,
) -> Option<ecdsa_recovery::PublicKey> {
    let puzzle_parsed = der::parse_der_sig(&hit.sig_puzzle)?;
    let puzzle_script_code = find_and_delete(full_script, &hit.sig_puzzle);
    let puzzle_digest = tx
        .legacy_sighash(input_index, &puzzle_script_code, puzzle_parsed.sighash_type)
        .ok()?;
    let (key_puzzle, _) = ecdsa_recovery::recover_first_pubkey(&puzzle_parsed, puzzle_digest).ok()?;
    Some(key_puzzle)
}
