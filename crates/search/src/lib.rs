use der::ParsedDerSig;
use puzzle::{PuzzleHit, SearchMode, evaluate_puzzle};
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
    /// The search space for (sequence, locktime) pairs.
    pub search_space: PinningSearchSpace,
    /// Whether to use strict DER or the easy-test predicate.
    pub mode: SearchMode,
    /// Which input index to compute the sighash for.
    pub input_index: usize,
}

/// Defines the (sequence, locktime) search space for pinning.
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
    /// The selected subset indices (signed + bonus).
    pub indices: Vec<usize>,
    /// The puzzle hit for this subset.
    pub puzzle_hit: PuzzleHit,
}

/// Search for a pinning solution by iterating over (sequence, locktime) pairs.
///
/// For each candidate, computes the sighash using the pinning script code and
/// evaluates the hash-to-sig puzzle. Returns the first hit, or `None` if the
/// search space is exhausted.
pub fn search_pinning(params: PinningSearchParams<'_>) -> Option<PinningHit> {
    let ss = &params.search_space;

    for seq_offset in 0..ss.sequence_count {
        let sequence = ss.sequence_start.wrapping_add(seq_offset);
        for lt_offset in 0..ss.locktime_count {
            let locktime = ss.locktime_start.wrapping_add(lt_offset);

            let mut tx = params.tx.clone();
            tx.inputs[params.input_index].sequence = sequence;
            tx.locktime = locktime;

            let digest = match tx.legacy_sighash(
                params.input_index,
                params.pin_script_code,
                params.sig_nonce.sighash_type,
            ) {
                Ok(d) => d,
                Err(_) => continue,
            };

            if let Some(hit) = evaluate_puzzle(params.sig_nonce, digest, params.mode) {
                return Some(PinningHit {
                    sequence,
                    locktime,
                    puzzle_hit: hit,
                });
            }
        }
    }

    None
}

/// Search for a digest solution by iterating over all C(n, t_total) subsets.
///
/// For each subset, applies FindAndDelete to remove the selected dummy signatures
/// and the round nonce signature from the full script, then computes the sighash
/// and evaluates the puzzle. Returns the first hit, or `None`.
pub fn search_digest(params: DigestSearchParams<'_>) -> Option<DigestHit> {
    let t_total = params.t_signed + params.t_bonus;

    // Remove the round nonce sig from the script (it's consumed by CHECKMULTISIG)
    let nonce_sig_bytes = der::encode_der_sig(
        &params.sig_nonce.r,
        &params.sig_nonce.s,
        params.sig_nonce.sighash_type,
    );
    let base_script_code = find_and_delete(params.full_script, &nonce_sig_bytes);

    for combo in CombinationIter::new(params.n, t_total) {
        // Apply FindAndDelete for each selected dummy sig
        let mut script_code = base_script_code.clone();
        for &index in &combo {
            script_code = find_and_delete(&script_code, &params.dummy_sigs[index]);
        }

        let digest = match params.tx.legacy_sighash(
            params.input_index,
            &script_code,
            params.sig_nonce.sighash_type,
        ) {
            Ok(d) => d,
            Err(_) => continue,
        };

        if let Some(hit) = evaluate_puzzle(params.sig_nonce, digest, params.mode) {
            return Some(DigestHit {
                indices: combo,
                puzzle_hit: hit,
            });
        }
    }

    None
}
