# Search space

This document describes what the off-chain search is actually
searching over, where the work budget comes from, and why each phase
is expensive.

## The three phases

A QSB spender runs three independent searches:

1. **Pinning**: find transaction parameters such that the
   recovered-key HASH160 is a valid DER signature.
2. **Round 1 digest**: find a C(n, t₁) subset of dummy signatures
   such that the resulting FindAndDelete-modified sighash passes the
   puzzle.
3. **Round 2 digest**: same as Round 1, with different nonce sig and
   different dummy pool.

Each phase has its own puzzle target (~2^46 hash-to-sig hits per
phase under the strict DER predicate). All three must succeed
before the spender can assemble a valid spending transaction.

## Phase 1: Pinning

### What varies

The spender enumerates transaction parameters that are committed to
by SIGHASH_ALL:

- `nLocktime` (4 bytes, 2^32 values)
- `nSequence` (4 bytes per input, 2^32 values per input)
- Output amounts (8 bytes per output)
- Output scripts (change address, OP_RETURN payload)
- Input ordering

In practice our implementation varies only **nLocktime** and
**nSequence** because both fields are adjacent to the end of the
sighash preimage — changing them doesn't affect the SHA-256 midstate
over the fixed prefix. This gives us a 2^64-element search space per
pinning attempt, which is astronomically larger than the ~2^46 needed
for a valid hit.

Other fields (output amounts, change address, OP_RETURN data) are
also valid search axes per the paper; they would force a larger
midstate suffix but the expanded search space is unnecessary given
how fast pinning already is.

### What the per-candidate work looks like

For each candidate:

1. Patch the 4 bytes of nSequence and 4 bytes of nLocktime in the
   sighash preimage suffix.
2. Finish SHA-256 from the precomputed midstate (1 block of work,
   since the suffix is ~68 bytes).
3. SHA-256 again (32 bytes) to get the sighash.
4. Recover `key_nonce = Recover(sig_nonce, sighash)` via the EC
   recovery formula.
5. Compute `HASH160(compressed_pubkey)`.
6. Check DER validity.

Cost: **1 SHA-256 block + 1 SHA-256(32) + EC recovery + HASH160**.
EC recovery is the dominant term; the SHA-256 work is negligible.

### Work budget

From the paper (section 4.3):

- Puzzle target per pinning attempt: **~2^46** candidates.
- Probability a single candidate is a hit: **~2^-46.4** for
  RIPEMD-160 under strict DER validation.
- Expected honest-spender work: **~2^46** hash-to-sig evaluations per
  attempt.

With Config A and the bonus-key optimization, the total digest
search space per attempt (C(150, 9) ≈ 2^46.2) is large enough that a
**single pinning attempt almost always produces a solvable
digest**. So the spender typically solves pinning exactly once per
UTXO.

### Measured throughput (Apple M4 Pro)

- Per-candidate GPU time: ~0.114 µs
- Batch throughput: **8.81 M candidates/sec** sustained
- Time to solve a 2^46 pinning attempt on one GPU: **~2,220 hours
  ≈ 92 days**

## Phase 2: Round 1 digest

### What varies

A subset `S ⊂ {0..n-1}` of size `t = 9` dummy signatures. Each
distinct subset produces a different `script_code` after
`FindAndDelete`, which produces a different sighash.

The search space is `C(150, 9) ≈ 2^46.2` subsets, enumerated by
index in lexicographic order. Each subset index maps uniquely to a
combination via `subset::nth_combination`.

### What the per-candidate work looks like

For each candidate (subset index):

1. Compute the subset `[i₀, i₁, ..., i₈]` via `nth_combination`
   (on the GPU, using a binomial coefficient table in device memory).
2. Sort the selected indices by their offset in `base_tail` (dummies
   are pushed in reverse order, so index order ≠ offset order).
3. Stream through `base_tail`, skipping the byte regions occupied by
   the selected dummies, feeding bytes into SHA-256 from the
   precomputed midstate.
4. Apply SHA-256 padding and finalize the first hash.
5. Second SHA-256 to produce the sighash.
6. EC recovery, HASH160, DER check — identical to pinning.

Round 1 `base_tail` is ~6,725 bytes, so the per-candidate SHA-256
work is **~102 compression invocations** — vastly more than pinning's
1 block. EC recovery and HASH160 are constant-cost, so Round 1's
cost is dominated by SHA-256.

### Work budget

- Puzzle target: ~2^46 per round (same predicate as pinning).
- Subset space: **C(150, 9) ≈ 2^46.2** with the t=9 bonus key variant.
- Expected honest-spender work: ~2^46.2 evaluations per round.

Both rounds are independent. Both must succeed. If either fails (no
valid subset in its respective 2^46.2 space), the spender re-pins
with a slightly different transaction and retries. The paper's
Config A puzzle target is tuned so that failure is rare.

### Measured throughput (Apple M4 Pro)

- Per-candidate GPU time: ~0.540 µs
- Batch throughput: **1.85 M subsets/sec** sustained
- Time to sweep 2^46.2 subsets on one GPU: **~12,100 hours ≈ 505 days**

## Phase 3: Round 2 digest

Structurally identical to Round 1, with two differences:

1. **Smaller `base_tail`** (~1,797 bytes). Round 2 sits at the end
   of the locking script, so the "tail" after the midstate boundary
   is much shorter than Round 1's (which has to include both the
   variable Round 1 dummies and the fixed Round 2 section).
2. **Fewer SHA-256 blocks per candidate** (~27 vs Round 1's ~102).

This makes Round 2 ~30% faster per candidate.

### Measured throughput (Apple M4 Pro)

- Per-candidate GPU time: ~0.418 µs
- Batch throughput: **2.39 M subsets/sec** sustained
- Time to sweep 2^46.2 subsets on one GPU: **~9,400 hours ≈ 392 days**

## Total Config A cost (single GPU)

| Phase | Target | Time |
|---|---|---|
| Pinning | 2^46 | ~2,220 hours |
| Round 1 | 2^46.2 | ~12,100 hours |
| Round 2 | 2^46.2 | ~9,400 hours |
| **Total** | | **~23,700 hours (~989 days)** |

### Multi-GPU scaling

The search is **embarrassingly parallel**. Each phase partitions
cleanly:

- **Pinning** — partition the locktime range. Worker `i` searches
  locktimes `[i·N, (i+1)·N)` for some large `N`.
- **Digest** — partition the combinatorial index range. Worker `i`
  searches `[i·N, (i+1)·N)` for some large `N`, computing its
  subsets via `nth_combination(start_index + gid)`.

No coordination between workers is needed until a hit is found; then
the first worker to find one can broadcast "done" and the others can
terminate.

With 10 GPUs: ~2,370 hours ≈ 99 days.
With 100 GPUs: ~237 hours ≈ 10 days.

Cloud cost at $0.50/hr/GPU: ~$12,000 regardless of GPU count
(linear scaling in work, constant cost per unit of work).

## Why the digest rounds dominate

Pinning is fastest because:

1. Its per-candidate SHA-256 work is tiny (1 block).
2. The search space is small (2^46).

Digest rounds are slower because:

1. Each candidate requires ~27-102 SHA-256 blocks.
2. The search space is larger by a factor of 2^0.2 (~15%).
3. `nth_combination` adds a small per-candidate cost (amortized
   across the 9 iterations of the O(t·n) algorithm).

At the end-to-end level, digest dominates ~9:1 in total cost —
exactly as the paper predicted in its cost-breakdown table (Section
4.6).

## The sighash midstate optimization

For all three phases the key structural trick is that **most of the
sighash preimage is the same across candidates**:

- Pinning: everything before the sequence field is fixed. Only ~68
  trailing bytes vary (sequence + outputs + locktime + sighash_type).
- Digest Round 1: everything before the first selectable Round 1
  dummy is fixed (~1,800 bytes). The tail (~6,725 bytes) contains
  the Round 1 dummies region + Round 2 section + outputs + etc.
- Digest Round 2: everything before the first selectable Round 2
  dummy is fixed (~6,700 bytes). The tail (~1,800 bytes) is the
  Round 2 section + outputs + etc.

The CPU pre-computes `sha256_midstate(fixed_prefix)` once per phase
and ships the 8 × u32 state to the GPU. Per-candidate SHA-256 work
then starts from the midstate and only processes the variable tail,
completing with standard padding.

This is why our pinning kernel is so fast — it does a single SHA-256
compression per candidate instead of hashing the full ~10 KB locking
script every time.

## What about bonus keys?

The QSB paper introduces an optimization called "bonus keys" to
bridge the gap between `C(150, 8) ≈ 2^42.3` (insufficient) and the
`2^46` puzzle target. Bonus keys are subset selections that
participate in FindAndDelete but skip HORS verification, so they
cost only 3 opcodes each instead of 9.

With `t_signed = 8, t_bonus = 1` (config A), the effective subset
count becomes `C(150, 9) ≈ 2^46.2`, slightly exceeding the puzzle
target and eliminating honest-spender grinding overhead.

Our implementation supports this via `QsbConfig::config_a()` — the
script builder knows how to emit the 3-opcode bonus selection loop.
Security implications (pre-image resistance drops from 2^138 to
2^118, collision resistance from 2^88 to 2^78) are discussed in the
paper's section 4.5.
