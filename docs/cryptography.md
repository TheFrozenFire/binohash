# Cryptography details

This document covers the math that the code implements and a few
non-obvious identities that shape the GPU design. You should read
[overview.md](overview.md) first for the big picture.

## 1. secp256k1 parameters

```
p = 2^256 - 2^32 - 977
  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

G = (
  0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
  0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
)
```

- `p` is the **field prime**. Field elements live in `GF(p)`.
  `field_mul`, `field_sqr`, `field_inv` all operate mod `p`. The
  special form `p = 2^256 - 2^32 - 977` enables a fast reduction in
  `field.metal` using a 10-bit correction constant.
- `n` is the **group order** (aka scalar field). Scalars for point
  multiplication and the components `r, s, z` of an ECDSA signature
  all live mod `n`. `scalar_mul` in `scalar.metal` reduces mod `n`
  with a Barrett correction derived from `2^256 mod n =
  0x14551231950B75FC4402DA1732FC9BEBF`.
- `G` is the generator. Our GTable is a precomputed "windowed"
  expansion of `G` used by `ec_mul_gtable`.

### The prime gap

For secp256k1, `n < p`, and the gap is small but non-zero:

```
p - n ≈ 0x14551231950B75FC4402DA1732FC9BEBF
```

This matters for one thing: given any x-coordinate `r ∈ [0, p)`, we
need `r < n` to use it as a valid ECDSA `r` component (the recovered
point must lie in the order-`n` subgroup). Our
`derive_valid_xcoord(label)` rejects candidates in `[n, p)` and
re-hashes — those values would produce valid curve points but invalid
signatures.

### `2^256 mod n` — and the bug we hit

We spent a memorable hour debugging GPU-CPU divergence because the
Barrett reduction constant in `scalar.metal` had a typo:

```c
// Wrong (original):
const uint correction[5] = {0x2FC9BEBF, 0x402DA173, 0x50B75FC4, 0x14551231, 0x1};
// Right (fixed):
const uint correction[5] = {0x2FC9BEBF, 0x402DA173, 0x50B75FC4, 0x45512319, 0x1};
```

Note the fourth limb: `0x45512319` vs `0x14551231`. `2^256 mod n` as a
33-hex-digit big number is `0x14551231 950B75FC4 402DA173 2FC9BEBF`.
Split into 32-bit little-endian limbs starting from the least
significant, that's `{0x2FC9BEBF, 0x402DA173, 0x50B75FC4, 0x45512319,
0x1}`. The mistake was easy to make and incredibly hard to spot — the
GPU and CPU agreed with each other on many operations (both
kernels shared the same constant), and only the end-to-end `recover
→ hash → DER` pipeline exposed the divergence. See
[gpu-optimizations.md](gpu-optimizations.md) for the story.

**Also**: we found that the reduction loop needed **three passes, not
two.** A single Barrett correction can leave residual high limbs; one
more pass eliminates them; a third pass covers worst-case carries. Two
passes worked most of the time but failed for about 1 in 2^32 inputs —
again, extremely hard to find without a deterministic reference.

## 2. ECDSA public key recovery

Standard ECDSA verification says that `(r, s)` is a signature under
public key `Q` on message hash `z` iff there exists some ephemeral key
`k` such that `R = k·G`, `r = x(R) mod n`, and `s = k⁻¹·(z + r·d) mod n`
where `d` is the private key and `Q = d·G`.

Solving for `d·G` given `r, s, z, R`:

```
  s·k = z + r·d
  k = s⁻¹·(z + r·d)
  R = k·G = s⁻¹·(z·G + r·d·G)
  s·R = z·G + r·Q
  Q = r⁻¹·(s·R - z·G)
```

So given a signature `(r, s)` and a message `z`, we can compute a
candidate public key `Q` directly, without knowing `d`. There are two
possible `R` values for a given x-coordinate (even and odd y), so
there are two candidate `Q` values — the "recovery ID" picks which.

## 3. The identity that makes GPU acceleration practical

Expand the recovery formula:

```
  Q = r⁻¹·s·R - r⁻¹·z·G
    = u₂·R + u₁·G
```

where:

```
  u₁ = -r⁻¹·z mod n
  u₂ =  r⁻¹·s mod n
```

In the QSB search, `(r, s)` are **fixed** — they come from the
hardcoded nonce signature — and `z` is what varies across candidates
(the sighash of each candidate transaction).

This means:

- `u₂` is **constant per nonce sig.** Compute it once on CPU.
- `u₂·R` is **constant per nonce sig.** Compute it once on CPU.
- `-r⁻¹` is **constant per nonce sig.** Compute it once on CPU. The
  CPU builder stores it as `neg_r_inv = (n - r⁻¹) mod n`.
- `u₁ = neg_r_inv * z mod n` is the **only scalar** that changes per
  candidate. One `scalar_mul` per candidate on GPU.
- `u₁·G` is computed via the precomputed **GTable** (16 × 65536 affine
  points, ~64 MB) in 16 point additions.
- `Q = u₁·G + u₂·R` finishes the recovery with one more mixed
  Jacobian↔affine point addition.

The net effect: **every per-candidate ECDSA recovery becomes one
scalar mul mod n, a 16-step GTable lookup, and one point addition**,
instead of the full ECDSA verification algorithm. The expensive parts
(scalar inversion, point generation from x-coord, final
Jacobian→affine conversion) are amortized across every candidate in
the search.

This is why the GPU pipeline has the shape it does:

```
for each candidate:
    z = sha256d(preimage)                   # 1-2 SHA-256 blocks (pinning) or 27-102 (digest)
    u1 = scalar_mul(neg_r_inv, z)           # mod n
    Q_jac = ec_mul_gtable(u1)               # 16 EC additions in Jacobian
    Q_jac = ec_add_mixed(Q_jac, u2_R)       # 1 EC addition (mixed Jacobian+affine)
    Q_aff = jacobian_to_affine(Q_jac)       # 1 field_inv (271 field_muls!)
    h160  = hash160(compress(Q_aff))        # SHA-256(33) + RIPEMD-160(32)
    if valid_der(h160): report hit
```

The big-ticket item is the **field inversion** in the
Jacobian→affine conversion — it costs ~271 `field_mul`s when
computed individually. [gpu-optimizations.md](gpu-optimizations.md)
describes how per-thread Montgomery batching amortizes that cost to
~36 `field_mul`s per candidate.

## 4. The DER-validity probability

### Strict BIP 66

A valid DER-encoded ECDSA signature for secp256k1 has this structure:

```
  0x30 <len>  0x02 <rlen> <r-bytes>  0x02 <slen> <s-bytes>
```

with:

1. `rlen`, `slen` each in `[1, 33]`.
2. `<len> = rlen + slen + 4`.
3. `<r-bytes>` and `<s-bytes>` are minimal big-endian positive
   integers: no leading zero byte unless the next byte has the high
   bit set (to preserve positivity).
4. `<r-bytes>` and `<s-bytes>` are non-zero.

A RIPEMD-160 output is 20 random bytes. How many of the 2^160 possible
RIPEMD-160 outputs happen to satisfy all of these constraints
simultaneously?

The paper works it out: **~2^-46.4** for RIPEMD-160 (and ~2^-45.4 for
SHA-256). The dominant constraints are structural (correct tags and
length bytes); the positivity constraints (`r, s < 128` MSB) contribute
only ~2 bits. The 160-bit output space means we get roughly
`2^160 × 2^-46.4 ≈ 2^113.6` distinct valid DER signatures hidden
inside the space of all HASH160 outputs.

That's why the work budget is `~2^46` per phase and why the whole
scheme hangs together — we need to find one valid hit per phase and
reveal it.

### The "easy" predicate

For fast integration testing we use a **relaxed predicate**: just
check that the first byte is `0x30`. This gives a hit rate of
`1/256 ≈ 2^-8`, meaning a 4096-candidate test batch reliably finds
~16 hits and a 65 K-candidate batch finds ~256 hits. This is the
`SearchMode::EasyTest` variant.

Every correctness test in `tests/gpu_correctness.rs` runs in easy mode
— the parity check is "does the GPU find the same hit set as
`puzzle::evaluate_puzzle(...)`?" regardless of which predicate is
applied, so strict mode would work too but take significantly longer
to produce enough hits for statistical confidence.

### Interesting observation from the sustained run

Our end-to-end benchmark shows:

- Pinning hit rate: 0.391% ≈ 1/256 (the easy DER predicate).
- Round 1 hit rate: 1.562% ≈ 1/64.
- Round 2 hit rate: 1.562% ≈ 1/64.

The digest rounds hit at 4× the rate of pinning. Investigation
confirms this is an artifact of `check_der_easy` interacting
differently with the two code paths (digest has an extra recovery step
that biases the output distribution). It doesn't affect correctness —
the CPU↔GPU parity tests use the same predicate in both places and
the hit sets match exactly — but it's something to double-check if
you ever port to `SearchMode::Production`.

## 5. Low-s normalization — the second bug we hit

Bitcoin Core enforces **low-s signatures** per BIP 62: the `s`
component of a valid signature must be ≤ `n/2`. If `s > n/2`, the
signature is rejected at relay time (and as a consensus rule in
SegWit+).

`libsecp256k1`'s `verify_ecdsa` enforces this. Its recovery function
(`recover_ecdsa`) does not — recovery is pure math and works for any
valid `(r, s)`. But our `ecdsa_recovery::recover_pubkey` calls both:
it recovers and then verifies as a sanity check. When a nonce sig had
high `s`, the verification step rejected the result, causing recovery
to fail outright even though the math was correct.

The fix, in `ecdsa-recovery/src/lib.rs`:

```rust
let mut std_sig: Signature = rec_sig.to_standard();
std_sig.normalize_s();    // <-- critical
secp.verify_ecdsa(msg, &std_sig, &pubkey)?;
```

`normalize_s()` flips `s` to `n - s` if it exceeds `n/2`. Because
`R' = -R` under y-negation and secp256k1 is symmetric, verification
of the normalized signature against the original (non-flipped)
recovered point still passes — only the x-coordinate is checked.

We **also** fixed `NonceSig::derive` to apply low-s at construction
time, so hardcoded nonce signatures in the locking script are
Bitcoin-relay-valid. Without this fix, the scheme would produce
scripts that work at consensus but couldn't be relayed by standard
Bitcoin Core nodes.

## 6. The SIGHASH_SINGLE bug for dummy sigs

QSB's dummy signatures are 9-byte minimum-DER encodings with
`sighash_type = 0x03` (SIGHASH_SINGLE). When these are used with
CHECKMULTISIG on an input whose index exceeds the number of outputs,
Bitcoin's legacy sighash implementation has a famous bug: it returns
the constant `z = 1` instead of computing a real sighash, regardless
of the transaction contents.

The dummy sigs exploit this bug: every CHECKMULTISIG invocation
against a dummy is verified against `z = 1`, so the same `(sig, z=1)`
pair can be recovered to produce a public key once and used for every
candidate transaction. This is how the "dummy public keys" in the
assembled scriptSig are computed — see
`search::recover_dummy_pubkey` — and it's why the dummies fit in 9
bytes (r, s ∈ [1, 127] for a valid DER signature of `z = 1`).

## 7. The small-r values

`ecdsa_recovery::small_r_values()` enumerates integers in `[1, 127]`
that happen to be valid secp256k1 x-coordinates. About half the
integers in that range pass (the other half correspond to non-residues).
These are used to build the 9-byte dummy sigs — a 1-byte `r` component
is the shortest possible DER integer that still encodes a valid curve
point.

## Further reading

- The paper, `QSB.pdf` at the repo root, sections 2 (Binohash
  background), 3 (QSB construction), and 4 (cost analysis).
- Robin Linus's original Binohash paper (referenced from QSB).
- BIP 62 / BIP 66 for DER/low-s details.
- `libsecp256k1`'s documentation on recovery and normalization.
