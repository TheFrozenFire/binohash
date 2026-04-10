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
}

impl NonceSig {
    /// Derive a nonce signature from a label string.
    ///
    /// Uses `derive_valid_xcoord("{label}_r")` for r and
    /// `derive_valid_scalar("{label}_s")` for s.
    pub fn derive(label: &str) -> Self {
        let r = derive_valid_xcoord(&format!("{label}_r"));
        let s = derive_valid_scalar(&format!("{label}_s"));
        let der_encoded = der::encode_der_sig(&r, &s, 0x01);
        Self { r, s, der_encoded }
    }

    /// Parse the DER-encoded signature into a `ParsedDerSig`.
    pub fn parsed(&self) -> der::ParsedDerSig {
        der::parse_der_sig(&self.der_encoded).expect("nonce sig should always be valid DER")
    }
}
