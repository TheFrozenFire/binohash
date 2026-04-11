use ecdsa_recovery::{derive_valid_scalar, derive_valid_xcoord, small_r_values};
use rand::RngCore;

/// HORS (Hash to Obtain Random Subset) key material.
///
/// Each entry consists of a secret preimage and its HASH160 commitment.
/// To "sign" a subset, the signer reveals the preimages at the selected indices.
/// The verifier checks each preimage against its commitment.
#[derive(Debug, Clone)]
pub struct HorsKeys {
    pub secrets: Vec<[u8; 20]>,
    pub commitments: Vec<[u8; 20]>,
}

impl HorsKeys {
    /// Generate `n` random HORS key pairs.
    pub fn generate(n: usize, rng: &mut impl RngCore) -> Self {
        let mut secrets = Vec::with_capacity(n);
        let mut commitments = Vec::with_capacity(n);
        for _ in 0..n {
            let mut secret = [0u8; 20];
            rng.fill_bytes(&mut secret);
            commitments.push(hash::hash160(&secret));
            secrets.push(secret);
        }
        Self {
            secrets,
            commitments,
        }
    }
}

/// Encode a 9-byte minimum DER dummy signature.
///
/// Format: `30 06 02 01 <r> 02 01 <s> 03` (sighash = SIGHASH_SINGLE).
/// Both `r` and `s` must be in [1, 127].
pub fn encode_minimal_dummy_sig(r: u8, s: u8) -> [u8; 9] {
    [0x30, 0x06, 0x02, 0x01, r, 0x02, 0x01, s, 0x03]
}

/// Generate `n` unique 9-byte dummy signatures for a given round.
///
/// Uses small r-values (valid secp256k1 x-coordinates in [1, 127]) paired with
/// s-values in [1, 127]. The round index offsets the enumeration to ensure
/// dummy sigs differ between rounds.
pub fn generate_dummy_sigs(n: usize, round: usize) -> Vec<[u8; 9]> {
    let small_rs = small_r_values();
    let mut sigs = Vec::with_capacity(n);
    for index in 0..n {
        let pair_index = index + round * n;
        let r = small_rs[pair_index % small_rs.len()];
        let s = 1 + ((pair_index / small_rs.len()) % 127) as u8;
        sigs.push(encode_minimal_dummy_sig(r, s));
    }
    sigs
}

/// A deterministically derived nonce signature used in the locking script.
///
/// The r component is a valid secp256k1 x-coordinate and s is a valid scalar,
/// both derived from a label string. The signature is encoded as DER with
/// SIGHASH_ALL (0x01).
#[derive(Debug, Clone)]
pub struct NonceSig {
    pub r: [u8; 32],
    pub s: [u8; 32],
    pub der_encoded: Vec<u8>,
    parsed: der::ParsedDerSig,
}

impl NonceSig {
    /// Derive a nonce signature from a label string.
    ///
    /// Uses `derive_valid_xcoord("{label}_r")` for r and
    /// `derive_valid_scalar("{label}_s")` for s.
    pub fn derive(label: &str) -> Self {
        let r = derive_valid_xcoord(&format!("{label}_r"));
        let s_raw = derive_valid_scalar(&format!("{label}_s"));
        // Enforce low-s (BIP 62): if s > N/2, use N - s.
        // This ensures the nonce sig is valid for Bitcoin's CHECKSIG.
        let s = Self::enforce_low_s(&s_raw);
        let der_encoded = der::encode_der_sig(&r, &s, 0x01);
        let parsed = der::ParsedDerSig {
            r,
            s,
            sighash_type: 0x01,
        };
        Self { r, s, der_encoded, parsed }
    }

    /// If s > N/2, return N - s (low-s normalization per BIP 62).
    fn enforce_low_s(s: &[u8; 32]) -> [u8; 32] {
        // N/2 (rounded down) = 7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0
        const N_HALF: [u8; 32] = [
            0x7F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0x5D, 0x57, 0x6E, 0x73, 0x57, 0xA4, 0x50, 0x1D,
            0xDF, 0xE9, 0x2F, 0x46, 0x68, 0x1B, 0x20, 0xA0,
        ];
        if s.iter().zip(N_HALF.iter()).find_map(|(a, b)| match a.cmp(b) {
            std::cmp::Ordering::Equal => None,
            ord => Some(ord),
        }).unwrap_or(std::cmp::Ordering::Equal) == std::cmp::Ordering::Greater {
            // s > N/2 → compute N - s
            let n = ecdsa_recovery::SECP256K1_N;
            let mut result = [0u8; 32];
            let mut borrow: u16 = 0;
            for i in (0..32).rev() {
                let diff = n[i] as u16 + 256 - s[i] as u16 - borrow;
                result[i] = diff as u8;
                borrow = if diff < 256 { 1 } else { 0 };
            }
            result
        } else {
            *s
        }
    }

    /// Get the parsed DER signature.
    pub fn parsed(&self) -> &der::ParsedDerSig {
        &self.parsed
    }
}
