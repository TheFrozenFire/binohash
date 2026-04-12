# GPU optimization journey

This document walks through every optimization we explored while
building the Metal GPU acceleration layer. It includes what worked,
what didn't, what we backed out, and what we judged not worth
pursuing. If you're reading this to learn "what changes should I
make to a CUDA port?", the order here is roughly "most impactful
first."

## The starting point

The initial kernel was a straightforward port: one thread per
candidate, each thread doing a full SHA-256 → scalar mul → EC
multiplication → Jacobian-to-affine conversion → HASH160 → DER
check. At 262K candidates per dispatch on an Apple M4 Pro this ran
in **47.7 ms**, which is 5.50 M candidates/sec.

Our GTable is 16 chunks × 65,536 entries × 32 bytes per coordinate
(both X and Y separately) = **64 MB** of precomputed windowed
multiples of `G`. This gets cached to disk (`gtable.bin`) because
recomputing it from scratch takes ~30-60 seconds.

Everything described below is measured against the 47.7 ms
baseline on an M4 Pro unless noted otherwise.

---

## Optimization 1: Fix the scalar_mul bug (!!!)

**What was wrong.** Our initial Barrett reduction constant for
`scalar_mul` mod `n` had a typo in the fourth limb:
`0x14551231` instead of `0x45512319`. The correct value is the
high limb of `2^256 mod n`.

**How we caught it.** The existing batch-pinning vs original-pinning
test passed — because both kernels shared the same `scalar_mul`
code, they agreed with each other. The only test that caught the
bug was the end-to-end GPU↔CPU parity test against
`ecdsa_recovery::recover_pubkey`. At that point the GPU had been
computing wrong public keys for weeks of development. The DER
"hits" it reported were real hits on the wrong pubkeys — which
looked statistically plausible because the DER predicate hits
~1/256 regardless of which key you HASH160.

**Second bug in the same function.** The Barrett reduction loop
only ran **two passes**. For the worst-case input, two passes can
leave residual high limbs after reduction. The third pass covers
those carry-out cases. Two passes also agreed between both kernels,
so again only the external reference caught it.

**Impact.** No performance change — just correct answers.

**Lesson.** Any self-consistent GPU pipeline can silently produce
wrong results. The only defense is to check the GPU's output
against an independent reference implementation on every primitive
you care about. Our `tests/gpu_correctness.rs` now has a test for
every layer: `field_mul`, `field_inv`, `scalar_mul`, `ec_mul`,
`ec_recovery`, pinning end-to-end, digest end-to-end, CPU↔GPU
parity at the search level.

---

## Optimization 2: Per-thread Montgomery batch inversion (**+62% throughput**)

**The bottleneck.** Our per-candidate field_inv (Jacobian→affine
conversion) costs ~271 `field_mul` operations per call — roughly
30% of the total pipeline time. At 262K candidates per batch, that's
~71 M `field_mul`s spent just on inversions.

**The trick.** Given N field elements `z[0], z[1], ..., z[N-1]`,
you can compute all N inverses with **one inversion and 3(N-1)
multiplications** using Montgomery's trick:

```
prod[0] = z[0]
for i in 1..N:
    prod[i] = prod[i-1] * z[i]

inv = field_inv(prod[N-1])     # <-- the only expensive inversion

for i in (N-1)..1:
    z_inv[i] = inv * prod[i-1]
    inv = inv * z[i]
z_inv[0] = inv
```

The cost per candidate drops from 271 to `271/N + 3(N-1)/N ≈ 36`
`field_mul`s at N=8 — an **87% reduction in inversion cost**, which
maps to ~25% of the original pipeline.

**Our first attempt (backed out).** We tried this with **cross-thread**
cooperation: threads within a threadgroup collaborated via
threadgroup memory and barriers to do a prefix-sum / Blelloch scan
of the Z values. Result: **slower than the baseline**, because
idle threads waited at barriers and wasted GPU execution units.
The Blelloch scan pattern also produces prefix-product inverses,
not element-wise inverses — we would have had to do the back-multiply
step in a second pass, making the scheme even worse.

**The fix (VanitySearch's approach).** Do the batching **per-thread,
entirely in registers.** Each thread processes N candidates
sequentially, accumulating its own Z values and running Montgomery's
trick on them without any cross-thread coordination.

**The catch: register pressure.** Each `uint256` is 8 × u32 = 32
bytes. Storing `jx[N], jy[N], jz[N], prod[N]` is `4 × 32 × N`
bytes per thread. At N=8, that's 1 KB per thread — already pushing
the Apple Silicon GPU register file. At N=16 we get a 72% speedup
but hit a rare correctness bug where one candidate per ~1000
produces a wrong answer — consistent with register spills
corrupting intermediate state. At N=32 the kernel regresses
outright because register spills to private memory dominate.

**Measured results** at 262K candidates:

| N | Time | Throughput | vs baseline |
|---|---|---|---|
| 1 (original) | 47.7 ms | 5.50 M/sec | — |
| 4 | 32.9 ms | 7.97 M/sec | +45% |
| **8** | **29.4 ms** | **8.92 M/sec** | **+62%** |
| 16 | 27.7 ms | 9.46 M/sec | +72% (rare errors) |
| 32 | 36.5 ms | 7.18 M/sec | +30% (register spills) |

We settled on **N=8** for production: correct, fast, well under the
register-spill threshold. The implementation is in
`kernels.metal::pinning_search_batched`.

**Why VanitySearch saw larger per-thread batching (GRP_SIZE=1024).**
Vanity search computes `base_point ± i·G` incrementally for
~1024 neighboring private keys and uses `_ModInvGrouped` to invert
all 513 Δx values at once. Their per-thread working set is
smaller because they only need the Δx values in the batch, not the
full Jacobian coordinates of every point. Our pipeline has a
fundamentally different shape (each candidate is an independent EC
recovery), so we can't reach N=1024 without register pressure.

---

## Optimization 3: GPU kernels for Round 1 and Round 2 digest search (**~35-62× CPU speedup**)

**The motivation.** Our pinning kernel already ran at 8.8 M/sec,
which takes ~2.2 hours to sweep the 2^46 pinning target. But the
paper estimates the digest rounds take **~5× more total work** than
pinning (more candidates + more SHA-256 per candidate), and without
GPU acceleration they would run entirely on the CPU at ~24-55K
subsets/sec — literally hundreds of days per round.

**The general-purpose digest kernel.** We designed **one kernel**
that handles both rounds (`digest_search` / `digest_search_nth`).
The kernel takes:

- `midstate`: SHA-256 state after the fixed prefix of the sighash
  preimage.
- `base_tail`: the tail bytes of the sighash preimage, with the
  round's nonce sig already `FindAndDelete`'d out but with all
  dummies still present.
- `dummy_offsets`: the position of each dummy's push within
  `base_tail`.
- A subset specification (either as a precomputed index buffer or
  as a start_index + on-GPU `nth_combination`).

Per candidate, the kernel:

1. Reads (or computes) this thread's subset of `t` indices.
2. Sorts by `dummy_offsets[subset[i]]` so skip regions are in
   ascending order (**critical gotcha**: dummies are pushed into the
   script in reverse order via `push_data(dummy[n-1..0])`, so higher
   indices have lower offsets — sorting by index produces the wrong
   skip order).
3. Streams through `base_tail`, skipping the byte regions of
   selected dummies, feeding bytes into SHA-256 resumed from the
   midstate.
4. Applies SHA-256 padding, finalizes the first hash.
5. Runs second SHA-256, EC recovery, HASH160, DER check — identical
   to the pinning kernel's back half.

**The varint gotcha.** The `script_code_len` varint in the sighash
preimage encodes the length of the *post-FindAndDelete* script, not
the pre-FindAndDelete length. Our Rust-side builder patches the
varint before computing the midstate so it reflects the final
post-FindAndDelete length. Without this patch, every candidate
computes a different preimage than the CPU and the GPU/CPU hit sets
don't match.

**The segment-based streaming optimization (+1.8%).** The original
implementation checked `if (read_pos == skip_starts[skip_idx])` on
every byte read. This is a branch per byte — inefficient on GPU.
The optimized version pre-computes segment boundaries
(`seg_start[0..t+1]`, `seg_end[0..t+1]`) and iterates
"for each segment, read bytes in bulk." The compiler generates
tighter code, avoiding the per-byte branch. Modest but free.

**The `constant` address space optimization (no measurable effect).**
We switched `base_tail` from `device` to `constant` on the theory
that the constant cache would speed up reads. No measured difference
— the L2 cache was already hitting for `base_tail` since every
thread reads the same ~1.8-6.7 KB region.

**Measured results** at 65,536 candidates, Config A (n=150, t=9):

| Phase | Time | Throughput | Paper CPU est. | Speedup |
|---|---|---|---|---|
| Round 2 (~27 SHA-256 blocks/candidate) | 33.5 ms | 1.97 M/sec | 55K/sec | **~36×** |
| Round 1 (~102 SHA-256 blocks/candidate) | 43.5 ms | 1.51 M/sec | 24K/sec | **~62×** |

**Why Round 1's speedup is larger than Round 2's.** Round 1 has
more SHA-256 work per candidate (102 vs 27 blocks). SHA-256 is
embarrassingly GPU-parallel, so adding SHA-256 work doesn't cost the
GPU much but costs the CPU linearly. The GPU's advantage grows with
Round 1's workload.

---

## Optimization 4: On-GPU `nth_combination` (**+33% end-to-end**)

**The problem.** The initial digest kernel took precomputed subset
indices as a buffer: the CPU computed `nth_combination(n, t, i)`
for each candidate in the batch and packed the results into a
`num_candidates × t` buffer. Measured CPU preprocessing cost:
**~16.2 ms per 65,536-subset batch** — ~48% of the GPU compute time
for Round 2. The CPU was the bottleneck.

**The fix.** Move `nth_combination` into the GPU kernel. Each thread
takes its combinatorial index `candidate_index = start_index + gid`
and walks the combinatorial number system algorithm to produce its
own subset indices on the fly.

**The implementation.** We upload a binomial coefficient table to
device memory (`binom_table[n][t+1]`, 150 × 10 × 8 bytes = 12 KB
for Config A) and implement the standard algorithm:

```c
ulong remaining = candidate_index;
uint next = 0;
uint remaining_k = t;
for (uint i = 0; i < t; i++) {
    uint c = next;
    while (true) {
        ulong count = binom_table[(n - 1 - c) * stride + (remaining_k - 1)];
        if (remaining < count) {
            subset[i] = c;
            next = c + 1;
            remaining_k--;
            break;
        }
        remaining -= count;
        c++;
    }
}
```

The inner while loop does up to ~17 iterations per output index
(the algorithm is O(t · n) worst case, but typical cases are faster
because the table lookups quickly narrow the range).

**Measured results** at 65,536 subsets, Round 2:

| Variant | GPU time | CPU preproc | Total |
|---|---|---|---|
| Precomputed subsets | 33.9 ms | 16.2 ms | ~50 ms |
| **nth_combination on GPU** | 33.5 ms | 0 ms | **33.5 ms** |

**33% end-to-end speedup** by eliminating the CPU bottleneck. The
GPU-side `nth_combination` is actually slightly faster than the
precomputed variant in isolation — the buffer upload of
`num_candidates × t × 4` bytes dominates the small cost of
computing the algorithm on the GPU.

**Memory transfer savings.** At 65K candidates × 9 indices × 4
bytes, the precomputed variant uploads 2.25 MB per batch. The
`nth` variant uploads a fixed 12 KB binom table once at kernel
dispatch, then nothing per batch.

This also makes **multi-GPU / distributed search trivial**: each
worker just gets a `start_index` and a `count` and computes its
own subsets. No pre-sharding required, no shared state beyond the
immutable parameter structs.

---

## Optimizations we explored but backed out

### Batch inversion via cross-thread cooperation (Blelloch scan)

Described above under "per-thread batch inversion." The
cross-thread variant was 1.47× **slower** than per-thread inversion
due to barrier overhead and idle threads. The parallel-prefix-scan
variant produced wrong results because it computes prefix-product
inverses, not element-wise inverses. Both backed out.

### 64-bit limb Montgomery multiplication

We implemented CIOS Montgomery multiplication with 4 × 64-bit
limbs. Carry propagation in MSL is painful — Metal doesn't have
inline PTX-style add-with-carry, so we had to manually track carry
via `ulong` overflow checks. We never got this implementation to
produce correct results for all inputs — intermediate accumulator
values exceeded `ulong` range in edge cases. **Abandoned.**

### 16-bit limb Montgomery multiplication

16 × 16-bit limbs would fit the accumulator in uint32 comfortably,
but for secp256k1 the P_0 limb (0xFFFFFC2F & 0xFFFF = 0xFC2F) is
close to 2^16, and `q = a*P_0 mod 2^16` can produce values where
`q * P + partial` exceeds 2^32. **Abandoned** in favor of...

### 13-bit limb Montgomery multiplication (kept but not integrated)

The `shaders/montgomery.metal` file implements a 20×13-bit CIOS
Montgomery multiplication that passes all correctness tests. It's
about 9% faster than our schoolbook `field_mul`. We never
integrated it into the production kernels because:

1. The speedup is small enough that the integration effort (porting
   every field operation to Montgomery form) exceeded the expected
   benefit.
2. The per-thread batch inversion optimization reduced the
   `field_inv` cost enough that field arithmetic stopped being the
   bottleneck.
3. Integrating Montgomery form adds conversion overhead at every
   kernel boundary (the GTable and scalar inputs are in standard
   form, Montgomery form requires an initial conversion and a
   final de-conversion).

The benchmark lives in `shaders/mont_benchmark_comparison.metal`
for anyone who wants to pick it up later.

### 20×13-bit limbs for EC operations (msl-secp256k1's approach)

We ported `msl-secp256k1`'s Jacobian addition code to Metal and
tried to use it instead of our own. **Pipeline creation failed**:
the 20-limb representation pushed per-thread register usage past
the Apple Silicon limit. Our 8×32-bit limb EC operations
compile without issue because the working set fits in the register
file. This was a useful data point — 13-bit limbs may be optimal
for CUDA on NVIDIA hardware but definitely aren't for Metal on
Apple Silicon.

### Endomorphism (GLV method)

secp256k1 has an efficient endomorphism `φ(x, y) = (β·x, y)` where
β is a cube root of unity mod p. Combined with a scalar
decomposition `k = k₁ + k₂·λ mod n`, this halves the scalar mul
cost. VanitySearch uses this to check 6 candidate pubkeys
(`Q, φ(Q), φ²(Q)` × `{y, -y}`) per EC multiplication.

**Why it doesn't work for us.** In QSB, the recovered public key
has a specific required shape: it must satisfy
`CHECKSIG(sig_nonce, key_nonce, sighash)`. If we substitute
`φ(key_nonce)`, the script's CHECKSIG fails — the substituted
point is a different public key and `sig_nonce` is not a valid
signature under it. Vanity address search is free to pick any
pubkey on the curve; QSB locks us to the specific one that
satisfies the signature equation.

The y-negation trick (flipping the 0x02/0x03 prefix) has the same
issue — negated pubkeys produce different HASH160 outputs, but
they also fail CHECKSIG.

### Incremental point addition (VanitySearch's GRP_SIZE trick)

VanitySearch computes `base_point ± i·G` for neighboring private
keys because consecutive keys differ by a known `G`. In QSB, the
u₁ values (`u₁ = -r⁻¹ · z_i`) are random because `z_i` comes from
SHA-256d. There's no arithmetic structure between consecutive
candidates — each needs a full scalar mul. **Not applicable.**

### GTable reconfiguration (8-bit windows, 12-bit windows)

We measured EC addition throughput at ~3.6 ns per addition on the
M4 Pro. Our current GTable uses 16-bit windows (16 chunks × 65,536
entries = 64 MB, 16 additions per scalar mul). Alternatives:

- **8-bit windows**: 32 chunks × 256 entries = 512 KB, 32 additions
  per scalar mul. Fits entirely in L2 cache, but 2× more work.
- **12-bit windows**: 22 chunks × 4096 entries = 5.6 MB, 22
  additions per scalar mul. Intermediate cache behavior.

Our 16-bit design is probably near-optimal for Apple Silicon's
large unified memory. The theoretical ceiling from halving the
number of additions is about 25% — but halving requires a 2^32-entry
table (256 GB). Smaller tables trade compute for cache, with
uncertain net benefit. **Not pursued.**

### Early DER rejection

Idea: check the first byte of the RIPEMD-160 output early and skip
the remainder of the pipeline if it's not `0x30`. **Impossible.**
RIPEMD-160 computes all 5 × 32-bit output words in the final round;
there's no "partial" output to inspect. And HASH160 is only ~2% of
the pipeline anyway, so even if we could skip it we'd save at most
~2%. **Not applicable.**

### CPU+GPU pipeline parallelism

Idea: while the GPU runs batch N, the CPU prepares batch N+1's
buffers and processes batch N-1's hits. **Not worth it.** After
moving `nth_combination` to the GPU, the CPU per-batch work is
~1 ms (buffer creation + result copying), dwarfed by the 33 ms
GPU time. Pipeline parallelism would save at most ~3% at the cost
of significant async refactoring. **Multi-GPU scaling (next) is
the better path to higher throughput.**

### Multi-GPU

**Trivial.** Each worker instantiates its own `MetalMiner` (one
per physical GPU) and processes a disjoint `[start_index, end_index)`
range. The digest search API already returns global u64 candidate
indices via `search_digest_batch_nth(start_index, count)`. The
pinning API is range-based via the `start_lt` parameter.

We haven't added explicit multi-GPU coordination because the test
machine has one GPU. The integration test
`gpu_digest_search_multi_worker_split` verifies that splitting a
range across 4 sequential invocations produces the same hit set as
a single contiguous run — which is how multi-GPU would actually
operate.

---

## What's left on the table

Concrete optimizations we didn't take that might still be worth
exploring:

- **Persistent Metal buffers.** The current per-batch dispatch
  creates ~10 fresh Metal buffers (params, base_tail,
  dummy_offsets, binom_table, hit_count, hit_indices, etc.). Most
  of these are constant across batches in a production loop. Caching
  them on the `MetalMiner` struct and only recreating per-batch
  output buffers could save ~1 ms per batch, which is a few percent
  at the sustained throughput level. This is the natural next
  optimization and the biggest remaining opportunity.
- **Vectorized `uint4` reads** from `base_tail` when segment
  alignment permits. The current streaming loop reads byte-by-byte;
  Metal's vectorized loads could potentially pull 16 bytes per
  memory transaction when aligned. Complex to implement because the
  skip regions are arbitrary byte offsets, but could reduce memory
  latency.
- **Unified kernel for pinning + digest** that dynamically selects
  between the two paths via a flag. Would reduce kernel launch
  overhead when running batches of mixed phases. Minor benefit for a
  lot of code complexity.
- **SHA-256 ARMv8 crypto extensions on CPU side.** For the CPU
  reference, the `sha2` crate uses ARMv8 SHA-256 instructions
  automatically on Apple Silicon. The CPU baseline in our benchmarks
  is already hitting those. Nothing to do.
