# binohash — Rust reference + GPU acceleration for QSB

A working Rust reference implementation of **QSB** (Quantum-Safe Bitcoin Transactions
Without Softforks) by Avihu Mordechai Levy, April 2026, together with a
GPU-accelerated off-chain search pipeline on Apple Silicon.

> **Status.** Research implementation. Everything is exercised by tests (146
> workspace tests, 8 fuzz targets) and a verified CPU↔GPU parity suite. Not
> reviewed for consensus-safety, side-channel resistance, or production use.

---

## What is this?

QSB is a scheme for producing Bitcoin transactions that remain secure even if
an adversary can break ECDSA (Shor's algorithm). It sits inside today's
Bitcoin script rules — no softfork, no new opcodes — and achieves ~118 bits of
second pre-image resistance under a quantum adversary, at the cost of a
non-trivial off-chain proof-of-work search that produces the spending
transaction.

This repository contains:

1. A **CPU reference implementation** of the full QSB scheme, split into
   single-purpose crates (hashing, DER, ECDSA recovery, HORS keys, transaction
   serialization, script, the hash-to-sig puzzle, subset enumeration, the
   search orchestrator).
2. A **Metal GPU acceleration layer** for the off-chain search, including
   kernels for all three production phases (pinning + Round 1 digest + Round 2
   digest) with on-GPU `nth_combination` and a CPU↔GPU parity test suite.
3. A deep documentation set (in `docs/`) explaining the scheme, the
   cryptographic building blocks, the search space, and every GPU optimization.

Both `QSB.pdf` (the paper being implemented) and a clone of the reference
`avihu28-qsb` Python implementation (in `ref/`) are vendored for study.

---

## Quick start

### Build and test

```bash
cargo test --workspace
```

Every crate has unit tests; `metal-gpu` has an integration test suite that
exercises each GPU primitive and every production kernel against the CPU
reference.

### Run the end-to-end pipeline throughput benchmark

```bash
cargo run --release -p metal-gpu --example pipeline_throughput
```

This measures sustained throughput of all three production phases
(pinning, Round 1 digest, Round 2 digest) for 20 seconds each using the
optimized kernels, then projects the total Config A search cost for a single
GPU.

Sample output on an Apple M4 Pro:

```
─── Phase 1: Pinning (batched N=8 kernel) ───
  Candidates: 176.42M | Hits: 689.2K (0.391%) | Rate: 8.81 M/sec

─── Phase 2: Round 1 digest (nth_combination kernel) ───
  Round 1 | Candidates: 37.09M | Hits: 579.6K (1.562%) | Rate: 1.85 M/sec

─── Phase 3: Round 2 digest (nth_combination kernel) ───
  Round 2 | Candidates: 47.78M | Hits: 746.5K (1.562%) | Rate: 2.39 M/sec
```

### Run the microbenchmarks

```bash
# All GPU microbenchmarks (warning: ~10 minutes)
cargo bench -p benches --bench gpu

# Targeted — specific kernel
cargo bench -p benches --bench gpu -- "GPU pinning batched"
```

### Fuzz

```bash
cargo +nightly fuzz run <target>     # e.g., der, scalar, sighash_crosscheck
```

Targets live under `fuzz/fuzz_targets/`.

---

## Performance summary (Apple M4 Pro)

| Phase | Throughput | vs CPU reference |
|---|---|---|
| Pinning (batched N=8 Montgomery) | 8.81 M candidates/sec | ~100× |
| Round 1 digest (~102 SHA-256 blocks/candidate) | 1.85 M subsets/sec | ~62× |
| Round 2 digest (~27 SHA-256 blocks/candidate) | 2.39 M subsets/sec | ~35× |

With the full Config A puzzle target (~2^46 per phase, 2 digest rounds), a
single M4 Pro takes roughly 1,000 GPU-hours end-to-end; multi-GPU scaling is
linear with no coordination beyond disjoint `start_index` ranges. See
[docs/benchmarks.md](docs/benchmarks.md) for the full numbers and
[docs/gpu-optimizations.md](docs/gpu-optimizations.md) for the journey that
produced them.

---

## Workspace layout

All crates live in `crates/`. Most are small, single-purpose libraries:

| Crate | Purpose |
|---|---|
| `hash` | SHA-256 / SHA-256d / RIPEMD-160 / HASH160, with `sha256_midstate` for partial-block state extraction |
| `der` | BIP66 strict DER validation, minimal DER encoding, and the `easy_der_predicate` used for fast integration testing |
| `ecdsa-recovery` | Public key recovery from `(r, s, z)` with low-s normalization, deterministic derivation of valid scalars and x-coordinates |
| `tx` | Minimal Bitcoin transaction types, varint encoding, legacy sighash (+ `legacy_sighash_preimage` for GPU midstate splitting) |
| `script` | Bitcoin Script building, `find_and_delete` (with the non-idempotent semantics required by Bitcoin Core), and the full QSB locking-script builder |
| `hors` | HORS key generation, minimal 9-byte DER dummy signature enumeration, and deterministic `NonceSig::derive` (with automatic BIP 62 low-s) |
| `puzzle` | The hash-to-sig predicate — recover `key_nonce = Recover(sig_nonce, z)`, check `HASH160(key_nonce)` is valid DER |
| `subset` | C(n,k) combinatorics: lexicographic iterator, `nth_combination` / `combination_index` for random access, safe `binomial_coefficient` |
| `search` | CPU orchestration of pinning search, digest search, chunked/resumable variants, and script-sig assembly |
| `metal-gpu` | Metal GPU acceleration: kernels, parameter builders, cached GTable, per-thread Montgomery batch inversion, on-GPU `nth_combination` |
| `benches` | Criterion microbenchmarks for primitives, CPU comparison, and GPU kernels |

Plus `fuzz/` (eight libFuzzer targets for `der`, `scalar`, `script`, `tx`,
`puzzle`, `subset`, `push_number`, and a CPU↔external sighash crosscheck) and
`ref/` (vendored upstream projects consulted during implementation).

---

## In-depth documentation

More detailed reading in the [`docs/`](docs/) directory:

- **[overview.md](docs/overview.md)** — The quantum threat to Bitcoin, the
  hash-to-sig puzzle, and how QSB assembles primitives into a quantum-safe
  script.
- **[architecture.md](docs/architecture.md)** — How the crates fit together,
  key types, data-flow diagrams, and where to start reading the code.
- **[search-space.md](docs/search-space.md)** — The three search phases
  (pinning, Round 1, Round 2), their work budgets, hit probabilities, and what
  determines the total off-chain cost.
- **[cryptography.md](docs/cryptography.md)** — secp256k1 mechanics, the
  recovery identity that lets us precompute `u2·R`, the DER predicate
  probability analysis, and the low-s nonce-sig gotcha we hit during
  development.
- **[gpu-optimizations.md](docs/gpu-optimizations.md)** — Full walk-through of
  every GPU optimization we explored: per-thread Montgomery batch inversion,
  on-GPU `nth_combination`, segment-based tail streaming, what we tried and
  backed out, and what isn't worth pursuing.
- **[benchmarks.md](docs/benchmarks.md)** — Numbers, numbers, numbers. Every
  measurement taken during optimization plus the Config A cost projection.

---

## Limitations and non-goals

- **Apple Silicon only for GPU acceleration.** The Metal code does not
  translate to CUDA/Vulkan/OpenCL. See
  [docs/gpu-optimizations.md](docs/gpu-optimizations.md) for notes on how the
  kernels map to a CUDA port if someone wants to do one.
- **Legacy (pre-SegWit) script only.** QSB fundamentally depends on legacy
  sighash + `FindAndDelete` semantics that were removed in BIP 143. This is
  not a bug — it is a constraint of the scheme.
- **Not reviewed for consensus policy or standardness.** QSB produces scripts
  that are consensus-valid but non-standard (bare >520-byte scripts). Relay
  requires a mining pool that accepts non-standard transactions.
- **Hash-to-sig puzzle only.** We do not implement the Lamport / HORS signing
  side of the spender workflow beyond what the script and search layers need;
  the secure-device side of the operational architecture is out of scope.
- **Easy-mode DER predicate is used for integration testing.** Production
  searches must set `SearchMode::Production`, which requires strict BIP 66
  DER validation (~2^-46 probability, not ~2^-8).

---

## Acknowledgements

- The QSB paper by Avihu Mordechai Levy (StarkWare), April 2026 — `QSB.pdf`
  in this repository.
- Robin Linus's **Binohash** construction (2026), which QSB modifies.
- [`geometryxyz/msl-secp256k1`](https://github.com/geometryxyz/msl-secp256k1)
  and [`zkmopro/gpu-acceleration`](https://github.com/zkmopro/gpu-acceleration)
  — studied for field-arithmetic and Montgomery-reduction ideas.
- [`JeanLucPons/VanitySearch`](https://github.com/JeanLucPons/VanitySearch) —
  the per-thread batch inversion strategy is directly inspired by its
  `_ModInvGrouped` pattern.
- [`JeanLucPons/Kangaroo`](https://github.com/JeanLucPons/Kangaroo) and
  [`brichard19/BitCrack`](https://github.com/brichard19/BitCrack) — additional
  reference points for GPU ECDLP / secp256k1 work.

## License

This repository contains research code. Consult `LICENSE` before using.
