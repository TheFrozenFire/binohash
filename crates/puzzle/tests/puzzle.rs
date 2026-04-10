use der::{encode_der_sig, parse_der_sig};
use ecdsa_recovery::{derive_valid_scalar, derive_valid_xcoord};
use puzzle::{SearchMode, check_hash_to_sig, evaluate_puzzle, try_recover_key_nonce};

fn make_test_nonce_sig() -> der::ParsedDerSig {
    let r = derive_valid_xcoord("qsb_pin_r");
    let s = derive_valid_scalar("qsb_pin_s");
    let encoded = encode_der_sig(&r, &s, 0x01);
    parse_der_sig(&encoded).expect("valid DER")
}

#[test]
fn try_recover_key_nonce_succeeds_for_valid_sig() {
    let nonce = make_test_nonce_sig();
    let digest = [0x42u8; 32];
    let result = try_recover_key_nonce(&nonce, digest);
    assert!(result.is_some(), "recovery should succeed for valid nonce sig");

    let (pubkey, recid) = result.unwrap();
    assert!(recid <= 1);
    let serialized = pubkey.serialize();
    assert_eq!(serialized.len(), 33);
}

#[test]
fn try_recover_produces_different_keys_for_different_digests() {
    let nonce = make_test_nonce_sig();
    let (key1, _) = try_recover_key_nonce(&nonce, [0x01; 32]).unwrap();
    let (key2, _) = try_recover_key_nonce(&nonce, [0x02; 32]).unwrap();
    assert_ne!(key1, key2, "different digests should yield different keys");
}

#[test]
fn check_hash_to_sig_returns_20_byte_hash() {
    let nonce = make_test_nonce_sig();
    let (pubkey, _) = try_recover_key_nonce(&nonce, [0x42; 32]).unwrap();
    let (sig_puzzle, _is_der) = check_hash_to_sig(&pubkey);
    assert_eq!(sig_puzzle.len(), 20);
}

#[test]
fn evaluate_puzzle_returns_none_for_typical_digest() {
    // The vast majority of digests will NOT produce a valid DER sig_puzzle
    // (~2^-46 probability). Test a few and confirm they're all None.
    let nonce = make_test_nonce_sig();
    for i in 0u32..100 {
        let mut digest = [0u8; 32];
        digest[..4].copy_from_slice(&i.to_le_bytes());
        let result = evaluate_puzzle(&nonce, digest, SearchMode::Production);
        assert!(
            result.is_none(),
            "digest {i} should not produce a DER hit in production mode"
        );
    }
}

#[test]
fn evaluate_puzzle_easy_mode_hits_more_often() {
    // Easy mode uses the relaxed predicate (~1/16 chance), so we should
    // get some hits in 100 attempts.
    let nonce = make_test_nonce_sig();
    let mut hits = 0;
    for i in 0u32..100 {
        let mut digest = [0u8; 32];
        digest[..4].copy_from_slice(&i.to_le_bytes());
        if evaluate_puzzle(&nonce, digest, SearchMode::EasyTest).is_some() {
            hits += 1;
        }
    }
    // With ~1/16 probability, we expect ~6 hits in 100 trials.
    // Allow a wide range to avoid flaky tests.
    assert!(
        hits > 0,
        "expected at least one easy-mode hit in 100 trials, got {hits}"
    );
}

#[test]
fn puzzle_hit_contains_valid_data() {
    let nonce = make_test_nonce_sig();
    // Find an easy-mode hit
    for i in 0u32..1000 {
        let mut digest = [0u8; 32];
        digest[..4].copy_from_slice(&i.to_le_bytes());
        if let Some(hit) = evaluate_puzzle(&nonce, digest, SearchMode::EasyTest) {
            // key_nonce should be a valid compressed pubkey
            let key_ser = hit.key_nonce.serialize();
            assert_eq!(key_ser.len(), 33);
            assert!(key_ser[0] == 0x02 || key_ser[0] == 0x03);

            // sig_puzzle should be 20 bytes
            assert_eq!(hit.sig_puzzle.len(), 20);

            // In easy mode, is_strict_der might be false
            // (the easy predicate is less strict)
            assert!(hit.recovery_id <= 1);
            return;
        }
    }
    panic!("expected at least one easy-mode hit in 1000 trials");
}

#[test]
fn production_hit_has_strict_der() {
    // We can't reliably find a production hit (2^46 work), but we can
    // verify that if we construct one synthetically, the flag is correct.
    // For now, just verify that production mode never returns a hit with
    // is_strict_der = false.
    let nonce = make_test_nonce_sig();
    for i in 0u32..100 {
        let mut digest = [0u8; 32];
        digest[..4].copy_from_slice(&i.to_le_bytes());
        if let Some(hit) = evaluate_puzzle(&nonce, digest, SearchMode::Production) {
            assert!(
                hit.is_strict_der,
                "production mode hits must have strict DER"
            );
        }
    }
}
