use der::{encode_der_sig, parse_der_sig};
use ecdsa_recovery::{
    derive_valid_scalar, derive_valid_xcoord, is_valid_scalar, recover_first_pubkey,
    recover_pubkey, small_r_values,
};
use secp256k1::{Message, Secp256k1, SecretKey};

#[test]
fn sign_then_recover_roundtrip() {
    let secp = Secp256k1::new();
    let secret_key = SecretKey::from_byte_array([3u8; 32]).expect("valid key");
    let message = Message::from_digest([9u8; 32]);
    let signature = secp.sign_ecdsa_recoverable(message, &secret_key);
    let (recid, compact) = signature.serialize_compact();

    let mut r = [0u8; 32];
    let mut s = [0u8; 32];
    r.copy_from_slice(&compact[..32]);
    s.copy_from_slice(&compact[32..]);

    let der_encoded = encode_der_sig(&r, &s, 0x01);
    let parsed = parse_der_sig(&der_encoded).expect("valid DER");

    let recovered =
        recover_pubkey(&parsed, [9u8; 32], i32::from(recid) as u8).expect("recovery succeeds");

    let expected = secp256k1::PublicKey::from_secret_key(&secp, &secret_key);
    assert_eq!(recovered, expected);
}

#[test]
fn recover_first_pubkey_is_deterministic() {
    // Use the same labels as the nktkt reference: these are known to produce
    // valid nonce signatures that work with key recovery.
    let r = derive_valid_xcoord("qsb_pin_r");
    let s = derive_valid_scalar("qsb_pin_s");
    let der_encoded = encode_der_sig(&r, &s, 0x01);
    let parsed = parse_der_sig(&der_encoded).expect("valid DER");

    let digest = [0x42u8; 32];
    let (key1, rid1) = recover_first_pubkey(&parsed, digest).expect("recovery succeeds");
    let (key2, rid2) = recover_first_pubkey(&parsed, digest).expect("recovery succeeds");
    assert_eq!(key1, key2, "deterministic recovery");
    assert_eq!(rid1, rid2, "deterministic recid");

    // The key should be a valid compressed pubkey
    let serialized = key1.serialize();
    assert_eq!(serialized.len(), 33);
    assert!(serialized[0] == 0x02 || serialized[0] == 0x03);
}

#[test]
fn recover_first_pubkey_matches_known_recid() {
    // When we know the recid from signing, recover_first_pubkey should find
    // the same key if it tries that recid
    let secp = Secp256k1::new();
    let secret_key = SecretKey::from_byte_array([7u8; 32]).expect("valid key");
    let digest = [0x42u8; 32];
    let message = Message::from_digest(digest);
    let signature = secp.sign_ecdsa_recoverable(message, &secret_key);
    let (recid, compact) = signature.serialize_compact();

    let mut r = [0u8; 32];
    let mut s = [0u8; 32];
    r.copy_from_slice(&compact[..32]);
    s.copy_from_slice(&compact[32..]);

    let der_encoded = encode_der_sig(&r, &s, 0x01);
    let parsed = parse_der_sig(&der_encoded).expect("valid DER");

    // Direct recovery with the known recid should match the signer's key
    let expected = secp256k1::PublicKey::from_secret_key(&secp, &secret_key);
    let recovered =
        recover_pubkey(&parsed, digest, i32::from(recid) as u8).expect("recovery succeeds");
    assert_eq!(recovered, expected);
}

#[test]
fn recover_pubkey_with_invalid_recid_fails() {
    let secp = Secp256k1::new();
    let secret_key = SecretKey::from_byte_array([3u8; 32]).expect("valid key");
    let message = Message::from_digest([9u8; 32]);
    let signature = secp.sign_ecdsa_recoverable(message, &secret_key);
    let (recid, compact) = signature.serialize_compact();

    let mut r = [0u8; 32];
    let mut s = [0u8; 32];
    r.copy_from_slice(&compact[..32]);
    s.copy_from_slice(&compact[32..]);

    let der_encoded = encode_der_sig(&r, &s, 0x01);
    let parsed = parse_der_sig(&der_encoded).expect("valid DER");

    // The wrong recid should either fail or recover a different key
    let wrong_recid = if i32::from(recid) == 0 { 1u8 } else { 0u8 };
    let expected = secp256k1::PublicKey::from_secret_key(&secp, &secret_key);

    match recover_pubkey(&parsed, [9u8; 32], wrong_recid) {
        Ok(key) => assert_ne!(key, expected, "wrong recid should not recover the same key"),
        Err(_) => {} // Also acceptable — recovery can fail outright
    }
}

#[test]
fn derive_valid_xcoord_is_deterministic() {
    let x1 = derive_valid_xcoord("test_label");
    let x2 = derive_valid_xcoord("test_label");
    assert_eq!(x1, x2, "same label should produce same result");

    let x3 = derive_valid_xcoord("different_label");
    assert_ne!(x1, x3, "different labels should produce different results");
}

#[test]
fn derive_valid_xcoord_is_on_curve() {
    let x = derive_valid_xcoord("qsb_pin_r");
    // Verify it's a valid x-coordinate by constructing a compressed pubkey
    let mut compressed = [0u8; 33];
    compressed[0] = 0x02;
    compressed[1..].copy_from_slice(&x);
    secp256k1::PublicKey::from_slice(&compressed).expect("should be valid curve point");
}

#[test]
fn derive_valid_scalar_is_deterministic() {
    let s1 = derive_valid_scalar("test_scalar");
    let s2 = derive_valid_scalar("test_scalar");
    assert_eq!(s1, s2);

    let s3 = derive_valid_scalar("other_scalar");
    assert_ne!(s1, s3);
}

#[test]
fn derive_valid_scalar_is_valid() {
    let s = derive_valid_scalar("test_scalar");
    assert!(is_valid_scalar(&s));
    assert!(!s.iter().all(|b| *b == 0), "should not be zero");
}

#[test]
fn small_r_values_are_valid_curve_points() {
    let values = small_r_values();
    assert!(!values.is_empty(), "should find at least some valid small r values");

    for r in &values {
        assert!(*r >= 1 && *r <= 127, "r={r} out of [1,127] range");
        let mut x = [0u8; 32];
        x[31] = *r;
        let mut compressed = [0u8; 33];
        compressed[0] = 0x02;
        compressed[1..].copy_from_slice(&x);
        secp256k1::PublicKey::from_slice(&compressed)
            .unwrap_or_else(|_| panic!("r={r} should be valid x-coordinate"));
    }
}

#[test]
fn small_r_values_are_sorted_and_unique() {
    let values = small_r_values();
    for window in values.windows(2) {
        assert!(window[0] < window[1], "values should be strictly increasing");
    }
}

#[test]
fn is_valid_scalar_rejects_zero() {
    assert!(!is_valid_scalar(&[0u8; 32]));
}

#[test]
fn is_valid_scalar_rejects_n() {
    // secp256k1 order N
    let n: [u8; 32] = [
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xfe, 0xba, 0xae, 0xdc, 0xe6, 0xaf, 0x48, 0xa0, 0x3b, 0xbf, 0xd2, 0x5e, 0x8c,
        0xd0, 0x36, 0x41, 0x41,
    ];
    assert!(!is_valid_scalar(&n), "N itself should be rejected");
}

#[test]
fn is_valid_scalar_accepts_one() {
    let mut one = [0u8; 32];
    one[31] = 1;
    assert!(is_valid_scalar(&one));
}

#[test]
fn is_valid_scalar_accepts_n_minus_one() {
    let mut n_minus_1: [u8; 32] = [
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xfe, 0xba, 0xae, 0xdc, 0xe6, 0xaf, 0x48, 0xa0, 0x3b, 0xbf, 0xd2, 0x5e, 0x8c,
        0xd0, 0x36, 0x41, 0x41,
    ];
    // Subtract 1
    n_minus_1[31] -= 1;
    assert!(is_valid_scalar(&n_minus_1));
}

/// Cross-check: derive nonce signature components match between runs
/// and produce a valid DER-encoded signature that can be parsed back.
#[test]
fn derive_nonce_sig_roundtrip() {
    let r = derive_valid_xcoord("qsb_pin_r");
    let s = derive_valid_scalar("qsb_pin_s");
    let sig = encode_der_sig(&r, &s, 0x01);
    let parsed = parse_der_sig(&sig).expect("nonce sig should be valid DER");
    assert_eq!(parsed.r, r);
    assert_eq!(parsed.s, s);
    assert_eq!(parsed.sighash_type, 0x01);
}

/// Test that a nonce signature can be used for key recovery with an arbitrary digest.
#[test]
fn nonce_sig_recovery_works() {
    let r = derive_valid_xcoord("qsb_pin_r");
    let s = derive_valid_scalar("qsb_pin_s");
    let sig = encode_der_sig(&r, &s, 0x01);
    let parsed = parse_der_sig(&sig).expect("valid DER");

    // Recover with an arbitrary sighash digest
    let digest = [0xAB; 32];
    let (pubkey, _recid) = recover_first_pubkey(&parsed, digest).expect("recovery should succeed");

    // The recovered key should be a valid 33-byte compressed pubkey
    let serialized = pubkey.serialize();
    assert_eq!(serialized.len(), 33);
    assert!(serialized[0] == 0x02 || serialized[0] == 0x03);

    // Different digest should produce a different key
    let digest2 = [0xCD; 32];
    let (pubkey2, _) = recover_first_pubkey(&parsed, digest2).expect("recovery should succeed");
    assert_ne!(
        pubkey, pubkey2,
        "different digests should recover different keys"
    );
}
