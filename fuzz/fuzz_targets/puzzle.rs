#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 32 {
        return;
    }

    // Use the first 32 bytes as a digest, try to evaluate the puzzle
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&data[..32]);

    // Construct a deterministic nonce sig for fuzzing
    let r = ecdsa_recovery::derive_valid_xcoord("fuzz_r");
    let s = ecdsa_recovery::derive_valid_scalar("fuzz_s");
    let encoded = der::encode_der_sig(&r, &s, 0x01);
    let parsed = der::parse_der_sig(&encoded).unwrap();

    // Property 1: evaluate_puzzle should never panic
    let _ = puzzle::evaluate_puzzle(&parsed, digest, puzzle::SearchMode::Production);
    let _ = puzzle::evaluate_puzzle(&parsed, digest, puzzle::SearchMode::EasyTest);

    // Property 2: try_recover_key_nonce should never panic
    let _ = puzzle::try_recover_key_nonce(&parsed, digest);

    // Property 3: if recovery succeeds, check_hash_to_sig should never panic
    if let Some((key, _)) = puzzle::try_recover_key_nonce(&parsed, digest) {
        let (sig_puzzle, is_der) = puzzle::check_hash_to_sig(&key);
        assert_eq!(sig_puzzle.len(), 20);

        // Property 4: if is_der, then is_valid_der_sig should agree
        if is_der {
            assert!(der::is_valid_der_sig(&sig_puzzle));
        }
    }
});
