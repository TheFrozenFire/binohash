use der::ParsedDerSig;
use ecdsa_recovery::PublicKey;
use puzzle::{PuzzleHit, SearchMode, evaluate_puzzle};
use rayon::prelude::*;
use script::{find_and_delete, push_data, push_number};
use subset::{CombinationIter, first_combination, next_combination};
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
    /// Optional: additional transaction modifications per candidate.
    /// Called with (tx, candidate_offset) to vary outputs, OP_RETURN, etc.
    /// The modifier should be cheap and deterministic.
    pub tx_modifier: Option<&'a (dyn Fn(&mut Transaction, u64) + Sync)>,
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
        if let Some(modifier) = params.tx_modifier {
            modifier(&mut tx, offset);
        }

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
) -> Option<PublicKey> {
    let puzzle_parsed = der::parse_der_sig(&hit.sig_puzzle)?;
    let puzzle_script_code = find_and_delete(full_script, &hit.sig_puzzle);
    let puzzle_digest = tx
        .legacy_sighash(input_index, &puzzle_script_code, puzzle_parsed.sighash_type)
        .ok()?;
    let (key_puzzle, _) = ecdsa_recovery::recover_first_pubkey(&puzzle_parsed, puzzle_digest).ok()?;
    Some(key_puzzle)
}

// ============================================================
// Dummy pubkey recovery
// ============================================================

/// Recover the public key for a dummy signature using the SIGHASH_SINGLE bug (z=1).
///
/// Dummy sigs are verified with z=1 (the SIGHASH_SINGLE bug produces this hash
/// when input_index >= output count). The pubkey is recovered from (sig, z=1).
pub fn recover_dummy_pubkey(dummy_sig: &[u8; 9]) -> Option<PublicKey> {
    let parsed = der::parse_der_sig(dummy_sig)?;
    let mut digest = [0u8; 32];
    digest[31] = 1; // z = 1 (SIGHASH_SINGLE bug)
    let (pubkey, _) = ecdsa_recovery::recover_first_pubkey(&parsed, digest).ok()?;
    Some(pubkey)
}

// ============================================================
// ScriptSig assembly
// ============================================================

/// Parameters for assembling the QSB scriptSig (unlocking script).
pub struct AssemblyParams<'a> {
    /// Pinning puzzle hit.
    pub pinning: &'a PinningHit,
    /// Pinning key_puzzle (recovered from sig_puzzle).
    pub pin_key_puzzle: &'a PublicKey,
    /// Round 1 digest hit.
    pub round1: &'a DigestHit,
    /// Round 2 digest hit.
    pub round2: &'a DigestHit,
    /// Dummy signatures for round 1.
    pub round1_dummy_sigs: &'a [[u8; 9]],
    /// Dummy signatures for round 2.
    pub round2_dummy_sigs: &'a [[u8; 9]],
    /// HORS secrets for round 1.
    pub round1_hors_secrets: &'a [[u8; 20]],
    /// HORS secrets for round 2.
    pub round2_hors_secrets: &'a [[u8; 20]],
}

/// Assemble the QSB scriptSig from search results.
///
/// The scriptSig pushes data bottom-to-top onto the stack. The locking script
/// evaluates pinning first, then round 1, then round 2, so the witness is
/// ordered: `[round2 data] [round1 data] [pin_key_puzzle] [pin_key_nonce]`.
pub fn assemble_script_sig(params: &AssemblyParams<'_>) -> Option<Vec<u8>> {
    let mut sig = Vec::new();

    // Round 2 (evaluated last by the script, so pushed first onto stack)
    append_round_witness(
        &mut sig,
        params.round2,
        params.round2_dummy_sigs,
        params.round2_hors_secrets,
    )?;

    // Round 1
    append_round_witness(
        &mut sig,
        params.round1,
        params.round1_dummy_sigs,
        params.round1_hors_secrets,
    )?;

    // Pinning: key_puzzle, key_nonce (key_nonce on top)
    sig.extend_from_slice(&push_data(&params.pin_key_puzzle.serialize()));
    sig.extend_from_slice(&push_data(&params.pinning.puzzle_hit.key_nonce.serialize()));

    Some(sig)
}

fn append_round_witness(
    out: &mut Vec<u8>,
    hit: &DigestHit,
    dummy_sigs: &[[u8; 9]],
    hors_secrets: &[[u8; 20]],
) -> Option<()> {
    let key_puzzle = hit.key_puzzle.as_ref()?;

    // key_puzzle and key_nonce for this round
    out.extend_from_slice(&push_data(&key_puzzle.serialize()));
    out.extend_from_slice(&push_data(&hit.puzzle_hit.key_nonce.serialize()));

    // Dummy pubkeys (reversed) — recovered from selected dummy sigs with z=1
    for &index in hit.indices.iter().rev() {
        let pubkey = recover_dummy_pubkey(&dummy_sigs[index])?;
        out.extend_from_slice(&push_data(&pubkey.serialize()));
    }

    // HORS preimages for signed indices (reversed)
    for &index in hit.signed_indices.iter().rev() {
        out.extend_from_slice(&push_data(&hors_secrets[index]));
    }

    // Selected indices (reversed)
    for &index in hit.indices.iter().rev() {
        out.extend_from_slice(&push_number(index as i64));
    }

    Some(())
}

// ============================================================
// Resumable search
// ============================================================

/// Progress state for resuming a pinning search.
#[derive(Debug, Clone)]
pub struct PinningProgress {
    /// Next offset into the (sequence × locktime) search space.
    pub next_offset: u64,
    /// Total candidates checked so far.
    pub checked: u64,
    /// Whether the entire search space has been exhausted.
    pub exhausted: bool,
}

/// Result of a chunked pinning search.
pub struct PinningSearchResult {
    /// The hit, if found.
    pub hit: Option<PinningHit>,
    /// Updated progress state for resumption.
    pub progress: PinningProgress,
}

/// Search for a pinning solution with a budget and resumable progress.
///
/// Searches up to `budget` candidates starting from `progress.next_offset`.
/// Returns updated progress for subsequent calls.
pub fn search_pinning_chunked(
    params: &PinningSearchParams<'_>,
    progress: PinningProgress,
    budget: u64,
) -> PinningSearchResult {
    let ss = &params.search_space;
    let total = ss.sequence_count as u64 * ss.locktime_count as u64;
    let lt_count = ss.locktime_count as u64;

    if progress.exhausted || progress.next_offset >= total {
        return PinningSearchResult {
            hit: None,
            progress: PinningProgress {
                exhausted: true,
                ..progress
            },
        };
    }

    let end_offset = (progress.next_offset + budget).min(total);
    let range_start = progress.next_offset;

    let found = (range_start..end_offset)
        .into_par_iter()
        .find_map_first(|offset| {
            let sequence = ss.sequence_start.wrapping_add((offset / lt_count) as u32);
            let locktime = ss.locktime_start.wrapping_add((offset % lt_count) as u32);

            let mut tx = params.tx.clone();
            tx.inputs[params.input_index].sequence = sequence;
            tx.locktime = locktime;
            if let Some(modifier) = params.tx_modifier {
                modifier(&mut tx, offset);
            }

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
        });

    PinningSearchResult {
        hit: found,
        progress: PinningProgress {
            next_offset: end_offset,
            checked: progress.checked + (end_offset - range_start),
            exhausted: end_offset >= total,
        },
    }
}

/// Progress state for resuming a digest search.
#[derive(Debug, Clone)]
pub struct DigestProgress {
    /// The next combination to try, or None if exhausted.
    pub next_combo: Option<Vec<usize>>,
    /// Total candidates checked so far.
    pub checked: u64,
    /// Whether all combinations have been exhausted.
    pub exhausted: bool,
}

/// Result of a chunked digest search.
pub struct DigestSearchResult {
    /// The hit, if found.
    pub hit: Option<DigestHit>,
    /// Updated progress state for resumption.
    pub progress: DigestProgress,
}

/// Search for a digest solution with a budget and resumable progress.
pub fn search_digest_chunked(
    params: &DigestSearchParams<'_>,
    progress: DigestProgress,
    budget: u64,
) -> DigestSearchResult {
    if progress.exhausted {
        return DigestSearchResult {
            hit: None,
            progress,
        };
    }

    let t_total = params.t_signed + params.t_bonus;
    let base_script_code = find_and_delete(params.full_script, params.sig_nonce_bytes);

    let initial = progress
        .next_combo
        .or_else(|| first_combination(params.n, t_total));

    let mut current = initial;
    let mut checked = 0u64;

    while checked < budget {
        let combo = match current.take() {
            Some(c) => c,
            None => break,
        };

        // Evaluate this candidate
        let mut script_code = base_script_code.clone();
        for &index in &combo {
            script_code = find_and_delete(&script_code, &params.dummy_sigs[index]);
        }

        let next = next_combination(&combo, params.n);
        checked += 1;

        if let Ok(digest) = params.tx.legacy_sighash(
            params.input_index,
            &script_code,
            params.sig_nonce.sighash_type,
        ) {
            if let Some(hit) = evaluate_puzzle(params.sig_nonce, digest, params.mode) {
                let key_puzzle = if hit.is_strict_der {
                    recover_key_puzzle(&hit, params.tx, params.full_script, params.input_index)
                } else {
                    None
                };

                let signed_indices = combo[..params.t_signed].to_vec();
                let bonus_indices = combo[params.t_signed..].to_vec();

                let exhausted = next.is_none();
                return DigestSearchResult {
                    hit: Some(DigestHit {
                        signed_indices,
                        bonus_indices,
                        indices: combo,
                        puzzle_hit: hit,
                        key_puzzle,
                    }),
                    progress: DigestProgress {
                        next_combo: next,
                        checked: progress.checked + checked,
                        exhausted,
                    },
                };
            }
        }

        current = next;
    }

    let exhausted = current.is_none();
    DigestSearchResult {
        hit: None,
        progress: DigestProgress {
            next_combo: current,
            checked: progress.checked + checked,
            exhausted,
        },
    }
}
