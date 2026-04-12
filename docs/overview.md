# Overview — the problem QSB solves

This document is the high-level "why does any of this exist" explainer.
It covers the quantum threat to Bitcoin, the hash-to-sig puzzle that QSB
is built around, and how the pieces assemble into a quantum-safe
transaction that a current Bitcoin node will accept.

## 1. The quantum threat to Bitcoin

Every Bitcoin transaction's spending authority is bound by an ECDSA
signature (or, post-Taproot, a Schnorr signature). Both sit on the
hardness of the **elliptic curve discrete logarithm problem** (ECDLP) on
secp256k1. Shor's algorithm — given a sufficiently large fault-tolerant
quantum computer — solves ECDLP in polynomial time, which means:

- Given a public key, recover the private key.
- Given a signature on any message, forge a different signature on any
  other message under the same key.
- For Taproot outputs, every UTXO is trivially spendable, because the
  Taproot key path is always available, regardless of the script tree.

The paper's threat model is "Shor works, Grover works." Under that model
elliptic curve primitives break catastrophically; hash functions are
merely weakened by a square-root factor (SHA-256's 256-bit security
becomes ≈128-bit security against pre-image search; RIPEMD-160's 160-bit
security becomes ≈80-bit security). A 160-bit hash still has a
comfortable margin for pre-image resistance — this is the gap QSB
exploits.

## 2. The naive "spend from a hash" idea

Suppose a UTXO's spending condition is `knowledge of a preimage of h`.
A spender reveals the preimage in a transaction, the script verifies
`HASH(preimage) == h`, and the coins move. Hash-based, nothing to break.

The catastrophe: the moment the honest spender broadcasts that
transaction, the preimage is visible to every node in the relay network
and every miner. A quantum-equipped adversary — or just a rational
miner — can grab the preimage, construct a **different** transaction
`T'` sending those same coins to an address they control, sign `T'`
with the same preimage, and broadcast `T'` *instead*. The adversary
doesn't have to break the hash. They just have to outcompete the honest
spender for block inclusion.

The problem reduces to this: **the preimage must commit to the exact
spending transaction, so that revealing it only authorizes that
transaction and no other.**

## 3. Binohash: committing a preimage to a transaction

Robin Linus's **Binohash** construction (2026) solved this by combining
two Lamport-signable proofs of work inside a legacy Bitcoin script:

1. A **pinning puzzle**. The spender grinds transaction parameters
   (nLocktime, nSequence, output amounts, OP_RETURN data, etc.) until
   the resulting transaction satisfies a proof-of-work check on some
   function of the sighash. Any modification to the transaction
   invalidates this work, forcing a full regrind.
2. A **digest puzzle**. Embedded in the locking script is a pool of `n`
   signatures and `n` HORS hash commitments. The spender selects a
   subset of `t` of them as "the digest"; different subsets produce
   different sighashes via Bitcoin's `FindAndDelete` mechanism. Only
   one subset satisfies a second proof-of-work check.

Because the selected subset is hashed into a **HORS** (Hash to Obtain
Random Subset) preimage set — which is post-quantum secure — Binohash's
transaction binding is quantum-safe. The attacker can't reveal a
different subset without performing a new Lamport signature, which
requires hash preimages they don't have.

Binohash, however, relied on a **signature-size-based** proof-of-work
check (OP_SIZE + OP_LESSTHAN): the spender searches for an ECDSA
signature whose DER encoding is unusually short. The security of that
check depends on the smallest known `r` value (x-coordinate of some
curve point with a known discrete log). This is exactly the kind of
assumption Shor destroys. An adversary with Shor could compute the
discrete log of `x=1`, producing a tiny `r`, producing short signatures
that satisfy the PoW for zero work, breaking the scheme.

## 4. QSB: swap the PoW puzzle for a hash-to-sig puzzle

QSB's contribution is to replace the signature-size PoW with a
**hash-to-sig puzzle**:

> **Hash-to-sig observation.** A random 20-byte string is, with
> probability ~2^-46, a valid DER-encoded ECDSA signature.

Why? DER signatures have a rigid structure:

```
0x30 <total-length> 0x02 <r-length> <r> 0x02 <s-length> <s>
```

The first byte must be 0x30. The length bytes must be internally
consistent with the total length. The integer tags 0x02 must appear at
the correct positions. Both `r` and `s` must have MSB < 128 (positive
integer) with no superfluous leading zero. Plug in a RIPEMD-160 output
(20 random bytes) and ~1 in 2^46 of them happen to satisfy all these
constraints simultaneously.

This gives QSB a new type of puzzle: **grind until some input's
RIPEMD-160 output is itself a valid DER signature.** The security
depends only on the pre-image resistance of RIPEMD-160 — not on any
elliptic curve assumption. Shor provides no advantage; Grover gives a
square-root speedup to ~2^23 effective work for the puzzle alone and
~2^69 for second pre-image attacks.

The clever part is how this gets verified *inside* a Bitcoin script:

1. The script hardcodes a nonce signature `sig_nonce` with a fixed
   SIGHASH_ALL flag.
2. The spender supplies `key_nonce = Recover(sig_nonce, sighash)` in
   the witness. The script verifies this via OP_CHECKSIGVERIFY —
   because `sig_nonce` is a valid signature under `key_nonce` iff
   `key_nonce` was derived from the current transaction's sighash, we
   get a transaction-bound public key "for free."
3. The script computes `sig_puzzle = RIPEMD160(key_nonce)` and treats
   the 20-byte output as if it were a DER signature.
4. The spender supplies `key_puzzle` in the witness. The script
   verifies `(sig_puzzle, key_puzzle)` is a valid ECDSA signature via
   a second OP_CHECKSIGVERIFY. This only passes if the RIPEMD-160 output
   is itself valid DER — i.e., the spender solved the hash-to-sig
   puzzle.

All verification is a handful of existing opcodes
(OVER, CHECKSIGVERIFY, RIPEMD160, SWAP, CHECKSIGVERIFY — five opcodes
for the pinning stage). No softfork, no new primitives. The resulting
scheme:

- **Quantum-safe**: depends only on pre-image resistance of RIPEMD-160.
- **Fits in legacy Bitcoin Script**: uses only consensus-valid
  pre-SegWit opcodes.
- **Script-size feasible**: ~9,650 bytes (under the 10,000 byte limit)
  for Config A, which achieves ~118 bits of pre-image security.

## 5. The cost: off-chain proof of work

QSB's quantum safety is not free. The spender must solve both puzzles
before broadcasting a transaction:

1. **Pinning**: grind transaction parameters until the derived sighash
   produces a nonce-recovered public key whose HASH160 is valid DER
   (~2^46 attempts).
2. **Digest** (two independent rounds): enumerate C(150,9) ≈ 2^46.2
   subsets of the dummy-signature pool; each FindAndDelete changes the
   sighash, and one subset per round must satisfy the puzzle.

The total work is roughly 2^46 + 2 × 2^46.2 ≈ 2^47.7 hash-to-sig
evaluations. On CPU this is weeks to months on a modern core. The
whole point of this repository is to push that onto a GPU so the
honest spender can actually produce a transaction in a day or two of
wall-clock time on commodity hardware.

## 6. What isn't quantum-safe

QSB does **not** protect existing Bitcoin UTXOs. It protects
transactions that *spend* from outputs whose spending condition is a
QSB-format script. Every previously-created Bitcoin output (standard
P2PKH, P2SH, SegWit, Taproot) remains vulnerable to Shor.

QSB also doesn't eliminate the sighash uncertainty problem that
Binohash addressed — it inherits the fix (hardcoding SIGHASH_ALL in the
nonce sig and FindAndDelete semantics), not the vulnerability.

## 7. What this codebase implements

The spender-side of the operational architecture:

- Build the locking script for a UTXO.
- Construct a candidate spending transaction template.
- Search for a valid pinning + digest solution (CPU and GPU).
- Assemble the scriptSig (the witness) once a solution is found.

We do **not** implement the secure-device key management side (the
HORS secret storage, the air-gap communication with the GPU farm) —
that is an engineering concern orthogonal to the cryptography.

## Next reading

- [architecture.md](architecture.md) — how the crates fit together.
- [cryptography.md](cryptography.md) — the actual math, including the
  identity that lets us precompute `u2·R` once per nonce sig.
- [search-space.md](search-space.md) — what work we do, how much, and
  where the hit probabilities come from.
- [gpu-optimizations.md](gpu-optimizations.md) — the journey from naive
  port to 8.8M pinning candidates/second.
