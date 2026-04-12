# Architecture

This document maps the codebase. Read it alongside a file tree — every
subsection corresponds to one crate and explains what it exports, what
depends on it, and where the interesting bits of code live.

## Crate dependency graph

```
             hash ←──────────── tx ←───┐
              ↑      ↗         ↑       │
              │    script ─────┤       │
              │       ↑        │       │
              │       │        │       │
              │       ├──── hors ──────┤
              │       │        ↑       │
              │       │        │       │
        ecdsa-recovery ← der   │       │
              ↑       ↑         ╲      │
              │       │          ╲     │
              └── puzzle ─────────→ search ──┐
                              │       ↑     │
                              │       │     │
                              └──  subset   │
                                            │
                                            ↓
                                       metal-gpu

                                          ↑
                                          │
                                       benches
                                       (dev-dep)
```

Every crate is `edition = "2024"`. Dependencies flow one way; there
are no cycles. Only `metal-gpu` directly pulls in the `metal` and
`secp256k1` crates — everything else is pure Rust with minimal
external dependencies (`sha2`, `ripemd`, `thiserror`, `rayon`).

---

## `hash` — hash primitives

**Exports.** `sha256`, `sha256d`, `ripemd160`, `hash160`,
`sha256_midstate`.

The first four are thin wrappers around the `sha2` / `ripemd` crates.
The interesting one is `sha256_midstate(data)`, which processes all
*complete* 64-byte blocks of `data` and returns the 8 × u32 SHA-256
state, discarding any trailing bytes. This is the crux of the
GPU's pinning optimization: the CPU pre-hashes the fixed prefix of the
sighash preimage once, passes the 32-byte state to the GPU, and the
GPU resumes from that midstate for each candidate (processing only
the varying suffix).

We re-implemented the SHA-256 compression function locally rather than
using `sha2::compress256` to avoid a `GenericArray` dependency chain,
and because we need exact control over "process complete blocks,
ignore the tail" semantics that differ from the standard final-block
padding.

---

## `der` — DER signature validation and the easy predicate

**Exports.** `encode_der_sig`, `parse_der_sig`, `ParsedDerSig`,
`is_valid_der_sig`, `easy_der_predicate`.

`is_valid_der_sig(bytes)` is the **strict BIP 66** check: it enforces
the complete DER structural constraints (compound header, integer tags
and lengths, minimal encoding, positive r/s, no leading zeros). This is
the predicate used in `SearchMode::Production`.

`easy_der_predicate(bytes)` is the integration-testing stand-in: it
checks only that the first byte is `0x30` (i.e., ~1/256 pass rate
instead of ~2^-46). With this predicate the test suite finds dozens to
hundreds of "hits" in a 4-8K-candidate batch, enough to exercise the
pipeline deterministically.

`encode_der_sig(r, s, sighash_type)` produces a minimum-size DER
encoding for a concrete `(r, s)` — used for building the hardcoded
`sig_nonce` values in the locking script.

---

## `ecdsa-recovery` — public key recovery + helpers

**Exports.** `recover_pubkey`, `recover_first_pubkey`,
`derive_valid_xcoord`, `derive_valid_scalar`, `small_r_values`,
`is_valid_scalar`, `SECP256K1_N`, `PublicKey` (re-exported from
`secp256k1`).

Wraps `libsecp256k1` via the `secp256k1` crate. The only subtlety is
**low-s normalization in `recover_pubkey`**: `libsecp256k1`'s
`verify_ecdsa` rejects high-s signatures (BIP 62), but pure recovery
doesn't require low-s. We normalize `s` inside the verification step so
that recovery still works even for nonce sigs that were not constructed
with low-s in mind. This was the first bug we hit integrating the GPU
kernel against the CPU reference — see
[cryptography.md](cryptography.md) for the full story.

`derive_valid_xcoord(label)` and `derive_valid_scalar(label)` are
deterministic "hash until valid" helpers: iterate
`SHA256(label || counter)` until the output is a valid scalar (< N) and
— for x-coords — a valid curve x-coordinate. These are used to build
reproducible `NonceSig` values for tests and benchmarks.

---

## `tx` — minimal Bitcoin transaction types

**Exports.** `Transaction`, `TxIn`, `TxOut`, `serialize_varint`,
`SIGHASH_*` constants, `legacy_sighash`, `legacy_sighash_preimage`.

A small Bitcoin transaction library with just the pieces QSB needs.
Only legacy (pre-SegWit) sighash — the whole scheme depends on
FindAndDelete, which was removed in BIP 143.

The fuzzing target `sighash_crosscheck.rs` runs our `legacy_sighash`
against the `bitcoin` crate's implementation for randomly generated
transactions and asserts they produce identical hashes. As of the
current commit we've had no divergences.

`legacy_sighash_preimage` returns the *unhashed* preimage bytes. This
is what the GPU consumes: we split the preimage at a 64-byte boundary,
feed everything before the boundary to `sha256_midstate` on the CPU,
and ship the midstate + suffix to the GPU.

---

## `script` — Bitcoin Script building + the QSB locking script

**Exports.** Opcode constants, `push_data`, `push_number`,
`find_and_delete`, `count_non_push_opcodes`, `validate_script_limits`,
`QsbConfig`, `build_pinning_script`, `build_round_script`,
`build_full_script`.

The *locking script builder*: given a pinning nonce sig, two round
nonce sigs, two rounds of HORS commitments, and two rounds of dummy
sigs, `build_full_script` emits the complete QSB script. Fits within
Bitcoin's 201-opcode / 10,000-byte consensus limits with Config A
(`n=150, t₁=9, t₂=9`).

`find_and_delete` is a **faithful re-implementation of Bitcoin Core's
non-idempotent semantics**. `F(F(s, x), x) != F(s, x)` in general —
specifically, removing `x` can create a new occurrence of `x` where
none existed before, which a second pass would then also remove. A
fuzzing target surfaced this during development; our tests codify the
correct behavior. Do not "fix" this in a refactor.

`QsbConfig::config_a` is the 118-bit-security production config
(`n=150, t₁=8+1b, t₂=7+2b`). `QsbConfig::test` is the tiny `n=20, t=2`
config used to keep integration tests fast.

---

## `hors` — HORS keys and nonce signatures

**Exports.** `HorsKeys`, `encode_minimal_dummy_sig`,
`generate_dummy_sigs`, `NonceSig`.

`HorsKeys::generate(n, rng)` creates `n` random 20-byte secrets and
their HASH160 commitments. A signer "signs" a subset by revealing the
preimages at the selected indices; the script verifies each preimage
against its commitment — this is the Lamport layer that makes the
digest quantum-safe.

`generate_dummy_sigs(n, round)` enumerates minimum-DER 9-byte
"signatures" using the small valid r-values that exist in `[1, 127]`
on secp256k1. These are unique per round; round 0's dummies are
disjoint from round 1's dummies by construction. Each dummy push is
exactly 10 bytes (`09` length prefix + 9 bytes of DER) — a constant
that shows up repeatedly in the GPU kernel design.

`NonceSig::derive(label)` deterministically constructs a `(r, s)` pair
from a label string, automatically applying low-s normalization so the
signature is valid under Bitcoin Core's CHECKSIG rules. The
`der_encoded` field contains the byte sequence pushed into the script.

---

## `puzzle` — the hash-to-sig predicate

**Exports.** `SearchMode`, `PuzzleHit`, `try_recover_key_nonce`,
`check_hash_to_sig`, `evaluate_puzzle`.

This is the crate that defines "what is a hit." `evaluate_puzzle`
takes a nonce sig, a sighash digest, and a `SearchMode`, and returns
`Some(PuzzleHit)` iff the recovered key's HASH160 is a valid DER
signature (strict mode) or passes the easy predicate (easy-test mode).

The GPU kernels reimplement this pipeline in Metal Shading Language;
`tests/gpu_correctness.rs` verifies that the GPU's hit set matches
`evaluate_puzzle` exactly across thousands of candidates.

---

## `subset` — combinatorics

**Exports.** `CombinationIter`, `first_combination`,
`next_combination`, `nth_combination`, `combination_index`,
`binomial_coefficient`.

The interesting one is `nth_combination(n, k, index)`. It returns the
`index`-th k-subset of `{0..n}` in lexicographic order in O(k*n) time
using the combinatorial number system. This gives us **random access**
into the digest search space, which is what makes multi-GPU /
distributed search easy: each worker just claims a non-overlapping
range `[start_index, end_index)` of combination indices, enumerates
them with `nth_combination`, and reports its hits.

The GPU kernel `digest_search_nth` reimplements this algorithm in
Metal and consults a device-memory binomial coefficient table.

`binomial_coefficient` uses `u128` with `checked_mul`, returning
`u128::MAX` on overflow. A fuzz target caught a panic here in an early
version — C(n, k) grows fast enough that unchecked multiplies explode.

---

## `search` — CPU orchestration

**Exports.** `PinningSearchParams`, `PinningSearchSpace`, `PinningHit`,
`DigestSearchParams`, `DigestHit`, `search_pinning`, `search_digest`,
`search_pinning_chunked`, `search_digest_chunked`,
`assemble_script_sig`, `recover_dummy_pubkey`.

Uses `rayon` to parallelize across CPU cores. `search_pinning` iterates
over `(sequence, locktime)` pairs and calls
`tx.legacy_sighash(input_index, pin_script_code, sighash_type)` for
each candidate; `search_digest` iterates over C(n, t) subsets, applying
`find_and_delete` per selection.

The `_chunked` variants take a `PinningProgress` / `DigestProgress`
value and a budget, returning an updated progress — they're the
building blocks for resumable searches that can checkpoint state
between runs.

`assemble_script_sig` constructs the unlocking witness once a solution
has been found: key_nonce, key_puzzle, revealed HORS preimages,
selected indices (reversed, because OP_ROLL is stack-based), dummy
pubkeys recovered via the SIGHASH_SINGLE bug. This is exactly the
"hot wallet builds the final witness" step in the operational
architecture.

---

## `metal-gpu` — Apple Silicon acceleration

**Exports.** `GpuError`, `GpuPinningHit`, `GpuSearchParams`,
`GpuDigestSearchParams`, `MetalMiner`.

The Metal work lives in three places:

- `shaders/*.metal` — the MSL kernels, split into single-purpose files
  that are concatenated at compile time (`uint256.metal`, `field.metal`,
  `scalar.metal`, `ec.metal`, `sha256.metal`, `ripemd160.metal`,
  `der.metal`, `kernels.metal`, plus a few benchmark-only files). Each
  file focuses on one layer of the pipeline.
- `src/gpu.rs` — the Rust wrapper around `metal`. `MetalMiner::new`
  compiles the shader, caches a 64 MB GTable, and stores persistent
  buffers. Per-batch methods (`search_pinning_batched`,
  `search_digest_batch_nth`, etc.) allocate the per-call buffers,
  dispatch the kernel, and collect results.
- `tests/gpu_correctness.rs` — 24 integration tests covering every
  primitive (SHA-256, HASH160, field_mul, field_inv, EC mul, EC
  recovery, Montgomery mul) and every production kernel against the
  CPU reference.

Key types:

- **`GpuSearchParams`** — precomputed values for pinning search:
  SHA-256 midstate, suffix, `neg_r_inv = -r⁻¹ mod N`, `u2·R` as an
  affine point, the locktime/sequence offsets within the suffix.
- **`GpuDigestSearchParams`** — analogous precomputed values for
  digest search: midstate, `base_tail` (everything after the midstate
  boundary, with the round's nonce sig already removed), dummy sig
  offsets within `base_tail`, scalar precomputations.

Both builders do CPU-side validation that the produced params round-trip
correctly (e.g., `sha256_midstate(prefix) + suffix_process == legacy_sighash`).
This is how we catch Config A edge cases without having to scale every
test up.

See [gpu-optimizations.md](gpu-optimizations.md) for a full walkthrough
of every optimization applied to the kernels.

---

## `benches` — Criterion microbenchmarks

**Exports.** None. Two benchmark binaries:

- `primitives.rs` — CPU-side primitive benchmarks (hash functions, DER
  parsing, script building, subset enumeration, legacy sighash).
- `gpu.rs` — every GPU kernel benchmark, plus `nth_combination` on CPU
  as a baseline, plus pinning-throughput comparisons across batch sizes.

Run with `cargo bench -p benches --bench gpu -- "<filter>"` — the
filter is a regex matched against benchmark names. Always use targeted
filters; running the whole GPU suite takes ~10 minutes.

---

## `fuzz/` — libFuzzer targets

Eight targets, one per source of potential consensus-critical bugs:

- **`der.rs`** — round-trip encode/parse for arbitrary inputs.
- **`scalar.rs`** — `is_valid_scalar` + `SECP256K1_N` comparisons.
- **`script.rs`** — `find_and_delete` and opcode counting.
- **`tx.rs`** — legacy sighash for arbitrary transactions.
- **`puzzle.rs`** — `evaluate_puzzle` with arbitrary digests.
- **`subset.rs`** — `nth_combination` and `combination_index`
  round-trip.
- **`push_number.rs`** — script number encoding edge cases.
- **`sighash_crosscheck.rs`** — our `legacy_sighash` vs the external
  `bitcoin` crate's implementation. The most important target: if the
  CPU reference diverges from Bitcoin Core's semantics, every downstream
  layer is wrong.

All targets currently run clean on standard corpora. The crosscheck
target found the non-idempotent FindAndDelete issue mentioned in
`script`.

---

## Key data flow: a spending transaction, end to end

1. **Setup (one-time).** Secure device generates HORS keys, sends
   commitments + two nonce sigs to the GPU farm. Commitments go into
   the locking script.
2. **Locking script construction.** Use `script::build_full_script` to
   assemble pinning + 2 rounds of opcodes, dummy sigs, HORS
   commitments. Validate it fits under 201 opcodes and 10 KB.
3. **Transaction template.** Build a candidate `tx` (version, input
   referring to the QSB UTXO, desired outputs, placeholder sequence and
   locktime).
4. **Pinning search.** Call
   `GpuSearchParams::from_pinning_search(...)` to derive the GPU
   params, then `MetalMiner::search_pinning_batched(...)` in a loop
   until a hit is found. Record `(sequence, locktime)` of the winning
   candidate and pin the transaction.
5. **Round 1 digest search.** Call
   `GpuDigestSearchParams::from_digest_search(...)` with the round 1
   nonce sig and dummies, then iterate
   `MetalMiner::search_digest_batch_nth(...)` over ranges of
   `start_index`. Record the winning subset index.
6. **Round 2 digest search.** Same as Round 1, but with the round 2
   nonce sig and dummies.
7. **Witness assembly.** On the secure device, verify the solution
   locally with `puzzle::evaluate_puzzle` (a single cheap check). If
   it passes, reveal the HORS preimages for the winning subsets and
   call `search::assemble_script_sig` to build the scriptSig.
8. **Broadcast.** Submit to a mining pool that accepts non-standard
   transactions (e.g., via Marathon's Slipstream).

The HORS secrets never leave the secure device. The GPU farm only ever
sees public data.

## Where to start reading the code

In priority order:

1. `crates/puzzle/src/lib.rs` (40 lines) — the single-screen definition
   of what "a hit" means.
2. `crates/search/src/lib.rs` — the CPU reference for the full search
   loop.
3. `crates/metal-gpu/shaders/kernels.metal` — every production kernel,
   annotated.
4. `crates/metal-gpu/src/gpu.rs` — the Rust ↔ Metal bridge and the
   parameter builders.
5. `crates/metal-gpu/tests/gpu_correctness.rs` — the correctness
   ground truth for every GPU claim in this repo.
