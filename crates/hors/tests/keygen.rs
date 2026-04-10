use hors::{HorsKeys, NonceSig, encode_minimal_dummy_sig, generate_dummy_sigs};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

#[test]
fn hors_keys_deterministic_with_seed() {
    let mut rng1 = ChaCha20Rng::seed_from_u64(42);
    let mut rng2 = ChaCha20Rng::seed_from_u64(42);

    let keys1 = HorsKeys::generate(10, &mut rng1);
    let keys2 = HorsKeys::generate(10, &mut rng2);

    assert_eq!(keys1.secrets, keys2.secrets);
    assert_eq!(keys1.commitments, keys2.commitments);
}

#[test]
fn hors_keys_different_seeds_differ() {
    let mut rng1 = ChaCha20Rng::seed_from_u64(42);
    let mut rng2 = ChaCha20Rng::seed_from_u64(99);

    let keys1 = HorsKeys::generate(10, &mut rng1);
    let keys2 = HorsKeys::generate(10, &mut rng2);

    assert_ne!(keys1.secrets, keys2.secrets);
}

#[test]
fn hors_commitments_verify() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let keys = HorsKeys::generate(50, &mut rng);

    for i in 0..50 {
        assert_eq!(
            hash::hash160(&keys.secrets[i]),
            keys.commitments[i],
            "commitment mismatch at index {i}"
        );
    }
}

#[test]
fn minimal_dummy_sig_is_9_bytes() {
    let sig = encode_minimal_dummy_sig(7, 42);
    assert_eq!(sig.len(), 9);
}

#[test]
fn minimal_dummy_sig_is_valid_der() {
    let sig = encode_minimal_dummy_sig(7, 42);
    assert!(der::is_valid_der_sig(&sig));
}

#[test]
fn minimal_dummy_sig_structure() {
    let sig = encode_minimal_dummy_sig(7, 42);
    assert_eq!(sig, [0x30, 0x06, 0x02, 0x01, 7, 0x02, 0x01, 42, 0x03]);
}

#[test]
fn generate_dummy_sigs_unique_within_round() {
    let sigs = generate_dummy_sigs(150, 0);
    assert_eq!(sigs.len(), 150);

    // Check uniqueness
    let mut seen = std::collections::HashSet::new();
    for sig in &sigs {
        assert!(seen.insert(*sig), "duplicate dummy sig found");
    }
}

#[test]
fn generate_dummy_sigs_different_rounds_differ() {
    let sigs0 = generate_dummy_sigs(150, 0);
    let sigs1 = generate_dummy_sigs(150, 1);
    assert_ne!(sigs0, sigs1, "different rounds should have different dummy sigs");
}

#[test]
fn all_dummy_sigs_are_valid_der() {
    for round in 0..2 {
        let sigs = generate_dummy_sigs(150, round);
        for (i, sig) in sigs.iter().enumerate() {
            assert!(
                der::is_valid_der_sig(sig),
                "round {round} index {i}: dummy sig is not valid DER"
            );
        }
    }
}

#[test]
fn nonce_sig_deterministic() {
    let sig1 = NonceSig::derive("qsb_pin");
    let sig2 = NonceSig::derive("qsb_pin");
    assert_eq!(sig1.r, sig2.r);
    assert_eq!(sig1.s, sig2.s);
    assert_eq!(sig1.der_encoded, sig2.der_encoded);
}

#[test]
fn nonce_sig_different_labels_differ() {
    let pin = NonceSig::derive("qsb_pin");
    let round1 = NonceSig::derive("qsb_round1");
    assert_ne!(pin.r, round1.r);
}

#[test]
fn nonce_sig_is_valid_der() {
    let sig = NonceSig::derive("qsb_pin");
    assert!(der::is_valid_der_sig(&sig.der_encoded));
}

#[test]
fn nonce_sig_parses_correctly() {
    let sig = NonceSig::derive("qsb_pin");
    let parsed = der::parse_der_sig(&sig.der_encoded).expect("should parse");
    assert_eq!(parsed.r, sig.r);
    assert_eq!(parsed.s, sig.s);
    assert_eq!(parsed.sighash_type, 0x01); // SIGHASH_ALL
}
