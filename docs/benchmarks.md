# Benchmarks

All numbers are from an Apple M4 Pro (12-core CPU, 16-core GPU,
36 GB unified memory, macOS). Anything labeled "per candidate" is
the total batch time divided by the batch size — includes buffer
allocation, kernel dispatch, and result readback.

## End-to-end pipeline throughput (sustained, 20 seconds per phase)

From `cargo run --release -p metal-gpu --example pipeline_throughput`:

| Phase | Kernel | Candidates | Hit rate | Rate |
|---|---|---|---|---|
| Pinning | `pinning_search_batched` (N=8 Montgomery) | 176.42 M | 0.391% | **8.81 M/sec** |
| Round 1 digest | `digest_search_nth` | 37.09 M | 1.562% | **1.85 M/sec** |
| Round 2 digest | `digest_search_nth` | 47.78 M | 1.562% | **2.39 M/sec** |

Config A parameters observed:

- Script size: 9,970 bytes (right at the 10 KB limit).
- Pinning preimage: 9,988 bytes; variable suffix: 68 bytes
  (1 SHA-256 block).
- Round 1 `base_tail`: 6,725 bytes (~105 blocks of variable SHA-256
  work per candidate).
- Round 2 `base_tail`: 1,797 bytes (~28 blocks of variable SHA-256
  work per candidate).

### Config A search cost (single M4 Pro)

| Phase | Target (Config A) | Wall time |
|---|---|---|
| Pinning | 2^46 candidates | ~2,220 hours (~92 days) |
| Round 1 | 2^46.2 subsets | ~12,100 hours (~505 days) |
| Round 2 | 2^46.2 subsets | ~9,400 hours (~392 days) |
| **Total** | | **~23,700 hours (~989 days)** |

Rough cloud cost estimate at $0.50/hr/GPU: **~$11,850**.

Multi-GPU scales linearly (no coordination beyond disjoint
`start_index` ranges):

- 10 GPUs: ~2,370 hours (~99 days) at roughly the same cost.
- 100 GPUs: ~237 hours (~10 days) at roughly the same cost.

### Sustained run: 1-hour pinning-only test

From the earlier `examples/throughput.rs` (60-minute sustained run
with the original `search_pinning_batch` kernel, prior to the
batched N=8 optimization):

| Metric | Value |
|---|---|
| Duration | 60 minutes |
| Total candidates | 19.78 B |
| Total hits | 77.28 M (0.39% hit rate) |
| Throughput | 5,495,000 candidates/sec (rock-solid, ±0.1% variation over 60 minutes) |
| Per-candidate | 0.182 µs |

This run confirmed no thermal throttling or drift across an hour
of continuous GPU load. The M4 Pro maintains full clocks
indefinitely for this workload.

## Pinning kernel variants (micro-benchmarks)

Criterion, 262,144 candidates per dispatch, real `GpuSearchParams`
from a Config A scenario:

| Variant | Time | Throughput | vs baseline |
|---|---|---|---|
| Original (one candidate per thread) | 47.7 ms | 5.50 M/sec | — |
| Batched N=4 | 32.9 ms | 7.97 M/sec | +45% |
| **Batched N=8 (production)** | **29.4 ms** | **8.92 M/sec** | **+62%** |
| Batched N=16 | 27.7 ms | 9.46 M/sec | +72% (rare correctness errors from register spills) |
| Batched N=32 | 36.5 ms | 7.18 M/sec | +30% (register pressure regression) |

The sustained benchmark reports 8.81 M/sec for the N=8 kernel,
slightly below the microbenchmark's 8.92 M/sec. The gap is
probably per-batch buffer allocation overhead that the microbench
amortizes across many iterations of the same batch.

## Digest kernel variants (micro-benchmarks)

Criterion, Config A (n=150, t=9), 65,536 subsets per dispatch:

| Variant | Time | Throughput |
|---|---|---|
| Round 2 precomputed subsets | 33.9 ms | 1.93 M/sec |
| Round 2 `digest_search_nth` | **33.5 ms** | **1.96 M/sec** |
| Round 1 precomputed subsets | 43.5 ms | 1.51 M/sec |
| Round 1 `digest_search_nth` | ~43 ms | ~1.52 M/sec |

CPU-side preprocessing cost (precomputed variants):

- `subset::nth_combination × 65,536 (n=150, t=9)`: **16.2 ms**.

So the end-to-end timings for the precomputed variants are
effectively `GPU time + 16.2 ms`, making them ~50% slower overall
than the `_nth` variants.

## GPU primitive throughputs

Criterion, 65,536 threads × 100 iterations unless noted:

| Operation | Throughput |
|---|---|
| `field_mul` (schoolbook 8 × 32-bit limbs) | ~6.6 G ops/sec |
| `field_sqr` | ~6.6 G ops/sec (currently just `field_mul(a, a)`) |
| `field_inv` (271-multiplication addition chain) | ~24 M ops/sec |
| `monty13_mul` (13 × 13-bit CIOS Montgomery) | ~7.2 G ops/sec |
| `monty13_sqr` | ~7.2 G ops/sec |
| `ref_mont_mul` (verbatim msl-secp256k1 algorithm) | ~6.8 G ops/sec |
| `hash160` (33-byte input) | ~1.2 G ops/sec |
| `ripemd160` (32-byte input, standalone) | ~1.9 G ops/sec |

Derived per-candidate cost estimate for the pinning pipeline
(before batch inversion):

- SHA-256: ~0.002 µs (negligible)
- scalar_mul mod n: small
- ec_mul_gtable: 16 × `ec_add_mixed` ≈ 16 × ~11 field_muls ≈ 176 muls
- ec_add_mixed (with u2r): ~11 muls
- field_inv: 271 muls (the big one, now amortized to ~36 via batch)
- HASH160: small

With N=8 batching, `field_inv` drops from 271 to ~36 amortized
`field_mul`s per candidate — roughly a 25% overall pipeline
improvement, which matches the observed +62% throughput (the
remainder comes from better GPU occupancy in the batched kernel).

## EC operation throughputs

From `bench_ec_comparison`:

| Kernel | 1 thread | 256 threads | 65,536 threads |
|---|---|---|---|
| `our_ec mixed_add` (8×32-bit Jacobian+Affine) | works | works | ~3.8 ms / 1 M additions |
| `geo_ec jac_add` (20×13-bit Jacobian+Jacobian) | **pipeline creation fails** | — | — |

The msl-secp256k1 20×13-bit Jacobian addition kernel **cannot be
compiled on Apple Silicon** — register pressure pushes the per-thread
register count past Metal's limit. Our 8×32-bit approach compiles
without issue. This is the most interesting negative result of the
whole project: **13-bit limbs are not universally optimal**; the
optimal limb width depends on the target GPU's register file size.

## CPU reference throughput

From `bench_cpu_puzzle_comparison`:

| Operation | Throughput |
|---|---|
| `puzzle::evaluate_puzzle` (single candidate, on CPU) | ~35,000 candidates/sec per core |
| `search::search_pinning` (12-core M4 Pro, rayon) | ~420,000 candidates/sec |

**GPU pinning vs CPU pinning: ~8.81 M / 0.42 M ≈ 21× speedup**
(versus a fully saturated 12-core CPU). The "100× speedup" often
quoted elsewhere is against a single CPU thread (~21× × 12 ≈ 252×,
discounted for rayon overhead).

## What these numbers do and don't tell you

These benchmarks are all:

- **Single GPU, Apple M4 Pro.** Performance will differ on M1/M2/M3
  or on non-Apple GPUs. In particular, Metal on M-series benefits
  heavily from unified memory (no PCIe transfer); a port to CUDA
  would need to think carefully about GTable placement.
- **Easy-mode DER predicate.** Production search uses
  `SearchMode::Production` (strict BIP 66 DER, ~2^-46 hit rate
  instead of ~2^-8). The per-candidate cost is **identical** in
  both modes — only the rate at which hits are reported changes.
- **Sustained / microbenchmark split.** Sustained runs include
  per-batch buffer allocation overhead; microbenchmarks don't. The
  sustained numbers are what you'd actually see in a production
  search loop; the microbenchmarks are "theoretical ceiling with
  perfect buffer reuse."

For a more thorough characterization on your hardware, run:

```bash
cargo run --release -p metal-gpu --example pipeline_throughput   # sustained
cargo bench -p benches --bench gpu -- "GPU pinning batched"      # micro
cargo bench -p benches --bench gpu -- "digest Round"             # micro
```
