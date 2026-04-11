use metal_gpu::{GpuSearchParams, MetalMiner};
use std::path::PathBuf;
use std::sync::LazyLock;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

static GTABLE_CACHE: LazyLock<PathBuf> = LazyLock::new(|| {
    let dir = std::env::temp_dir().join("binohash_test");
    std::fs::create_dir_all(&dir).ok();
    dir.join("gtable.bin")
});

fn miner() -> MetalMiner {
    MetalMiner::new(Some(&GTABLE_CACHE)).expect("Metal device and GTable should be available")
}

#[test]
fn gpu_sha256_matches_cpu() {
    let m = miner();
    println!("GPU device: {}", m.device_name());

    let input = [0x42u8; 32];
    let gpu_result = m.test_sha256(&input);
    let cpu_result = hash::sha256(&input);

    assert_eq!(
        hex(&gpu_result),
        hex(&cpu_result),
        "GPU SHA-256 should match CPU"
    );
}

#[test]
fn gpu_sha256_known_vector() {
    let m = miner();
    // SHA-256 of 32 zero bytes
    let input = [0u8; 32];
    let gpu_result = m.test_sha256(&input);
    let cpu_result = hash::sha256(&input);
    assert_eq!(hex(&gpu_result), hex(&cpu_result));
}

#[test]
fn gpu_hash160_matches_cpu() {
    let m = miner();

    // Test with a realistic compressed pubkey
    let mut input = [0u8; 33];
    input[0] = 0x02;
    input[1] = 0x79;
    input[2] = 0xBE;
    input[31] = 0x98;
    input[32] = 0x01;

    let gpu_result = m.test_hash160(&input);
    let cpu_result = hash::hash160(&input);

    assert_eq!(
        hex(&gpu_result),
        hex(&cpu_result),
        "GPU HASH160 should match CPU"
    );
}

#[test]
fn gpu_field_mul_known_values() {
    let m = miner();

    // a = 2, b = 3 → result should be 6
    let mut a = [0u8; 32];
    a[31] = 2;
    let mut b = [0u8; 32];
    b[31] = 3;

    let result = m.test_field_mul(&a, &b);
    let mut expected = [0u8; 32];
    expected[31] = 6;

    assert_eq!(hex(&result), hex(&expected), "2 * 3 mod P should be 6");
}

#[test]
fn gpu_field_mul_matches_python() {
    let m = miner();

    // Verify against Python: a*b mod P for known large values
    let mut a = [0u8; 32]; a[0] = 0x7F; a[31] = 0x42;
    let mut b = [0u8; 32]; b[0] = 0x3A; b[31] = 0x99;

    let result = m.test_field_mul(&a, &b);
    // Python: (0x7F00...42 * 0x3A00...99) mod P =
    let expected_hex = "aaa6000000000000000000000000000000000000000000001cc60135cfa922ba";
    assert_eq!(
        hex(&result), expected_hex,
        "GPU field_mul should match Python for large inputs"
    );
}

#[test]
fn gpu_field_mul_large_values() {
    let m = miner();

    // Use values that would overflow 256 bits without reduction
    // P - 1 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2E
    // (P-1) * 2 mod P = P - 2 (since (P-1)*2 = 2P - 2 ≡ -2 ≡ P-2)
    let p_minus_1: [u8; 32] = [
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
        0xFF, 0xFF, 0xFC, 0x2E,
    ];
    let mut two = [0u8; 32];
    two[31] = 2;

    let result = m.test_field_mul(&p_minus_1, &two);

    // Expected: P - 2
    let p_minus_2: [u8; 32] = [
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
        0xFF, 0xFF, 0xFC, 0x2D,
    ];

    assert_eq!(hex(&result), hex(&p_minus_2), "(P-1)*2 mod P should be P-2");
}

#[test]
fn gpu_field_inv_produces_identity() {
    let m = miner();

    // inv(a) should satisfy a * inv(a) = 1 mod P
    let mut a = [0u8; 32];
    a[31] = 42;

    let (_inv, product) = m.test_field_inv(&a);

    let mut one = [0u8; 32];
    one[31] = 1;

    assert_eq!(
        hex(&product),
        hex(&one),
        "a * inv(a) mod P should be 1"
    );
}

#[test]
fn gpu_field_inv_large_value() {
    let m = miner();

    // Test with a large value near P
    let a: [u8; 32] = [
        0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87,
        0x0B, 0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B,
        0x16, 0xF8, 0x17, 0x98,
    ]; // This is the x-coordinate of the secp256k1 generator

    let (_inv, product) = m.test_field_inv(&a);

    let mut one = [0u8; 32];
    one[31] = 1;

    assert_eq!(
        hex(&product),
        hex(&one),
        "G.x * inv(G.x) mod P should be 1"
    );
}

#[test]
fn gpu_ec_mul_generator() {
    let m = miner();

    // scalar = 1 → should produce the generator point G
    let mut scalar = [0u8; 32];
    scalar[31] = 1;

    let (x, y) = m.test_ec_mul(&scalar);

    // secp256k1 generator G coordinates (big-endian)
    let gx: [u8; 32] = [
        0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87,
        0x0B, 0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B,
        0x16, 0xF8, 0x17, 0x98,
    ];
    let gy: [u8; 32] = [
        0x48, 0x3A, 0xDA, 0x77, 0x26, 0xA3, 0xC4, 0x65, 0x5D, 0xA4, 0xFB, 0xFC, 0x0E, 0x11,
        0x08, 0xA8, 0xFD, 0x17, 0xB4, 0x48, 0xA6, 0x85, 0x54, 0x19, 0x9C, 0x47, 0xD0, 0x8F,
        0xFB, 0x10, 0xD4, 0xB8,
    ];

    assert_eq!(hex(&x), hex(&gx), "1*G x-coordinate should match");
    assert_eq!(hex(&y), hex(&gy), "1*G y-coordinate should match");
}

#[test]
fn gpu_ec_mul_matches_secp256k1_crate() {
    use secp256k1::{PublicKey, Secp256k1, SecretKey};

    let m = miner();
    let secp = Secp256k1::new();

    // Test with scalar = 12345
    let mut scalar = [0u8; 32];
    scalar[30] = 0x30;
    scalar[31] = 0x39; // 12345 = 0x3039

    let (gpu_x, gpu_y) = m.test_ec_mul(&scalar);

    let sk = SecretKey::from_byte_array(scalar).expect("valid");
    let pk = PublicKey::from_secret_key(&secp, &sk);
    let uncompressed = pk.serialize_uncompressed();
    let cpu_x = &uncompressed[1..33];
    let cpu_y = &uncompressed[33..65];

    assert_eq!(hex(&gpu_x), hex(cpu_x), "GPU EC mul x should match CPU for scalar 12345");
    assert_eq!(hex(&gpu_y), hex(cpu_y), "GPU EC mul y should match CPU for scalar 12345");
}

#[test]
fn gpu_ec_mul_matches_cpu_large_scalar() {
    use secp256k1::{PublicKey, Secp256k1, SecretKey};

    let m = miner();
    let secp = Secp256k1::new();

    // Test with a realistic scalar (derived from a label)
    let scalar = ecdsa_recovery::derive_valid_scalar("gpu_test_scalar");
    let (gpu_x, gpu_y) = m.test_ec_mul(&scalar);

    let sk = SecretKey::from_byte_array(scalar).expect("valid");
    let pk = PublicKey::from_secret_key(&secp, &sk);
    let uncompressed = pk.serialize_uncompressed();
    let cpu_x = &uncompressed[1..33];
    let cpu_y = &uncompressed[33..65];

    assert_eq!(
        hex(&gpu_x),
        hex(cpu_x),
        "GPU EC mul should match CPU for large scalar"
    );
    assert_eq!(hex(&gpu_y), hex(cpu_y));
}

#[test]
fn gpu_montgomery_mul_matches_schoolbook() {
    let m = miner();

    // Test several value pairs — Montgomery mul should produce identical results to schoolbook
    let test_cases: &[([u8; 32], [u8; 32])] = &[
        // Small values
        ({let mut a = [0u8;32]; a[31]=2; a}, {let mut b = [0u8;32]; b[31]=3; b}),
        // Large values
        ({let mut a = [0u8;32]; a[0]=0x7F; a[31]=0x42; a},
         {let mut b = [0u8;32]; b[0]=0x3A; b[31]=0x99; b}),
        // Generator x-coordinate
        (
            [0x79,0xBE,0x66,0x7E,0xF9,0xDC,0xBB,0xAC,0x55,0xA0,0x62,0x95,0xCE,0x87,
             0x0B,0x07,0x02,0x9B,0xFC,0xDB,0x2D,0xCE,0x28,0xD9,0x59,0xF2,0x81,0x5B,
             0x16,0xF8,0x17,0x98],
            [0x48,0x3A,0xDA,0x77,0x26,0xA3,0xC4,0x65,0x5D,0xA4,0xFB,0xFC,0x0E,0x11,
             0x08,0xA8,0xFD,0x17,0xB4,0x48,0xA6,0x85,0x54,0x19,0x9C,0x47,0xD0,0x8F,
             0xFB,0x10,0xD4,0xB8],
        ),
    ];

    for (i, (a, b)) in test_cases.iter().enumerate() {
        let schoolbook = m.test_field_mul(a, b);
        let montgomery = m.run_simple_kernel_2in_1out("test_monty_mul", a, b, 32);
        assert_eq!(
            hex(&schoolbook), hex(&montgomery),
            "Montgomery must match schoolbook for test case {i}"
        );
    }
}

#[test]
fn gpu_batch_pinning_matches_original() {
    let m = miner();

    // Use dummy but consistent parameters
    let nonce = hors::NonceSig::derive("gpu_batch_test");
    let parsed = nonce.parsed();
    let neg_r_inv = parsed.r; // Not mathematically correct, but consistent between kernels
    let u2r_x = parsed.r;
    let u2r_y = parsed.s;
    let suffix = vec![0u8; 32];
    let midstate = [0u32; 8];

    // Run original kernel — 1024 candidates in easy mode
    let original_hits = m.search_pinning_batch(
        &midstate, &suffix, 5000, 4, 8, 0xFFFFFFFE, 1, 1024,
        &neg_r_inv, &u2r_x, &u2r_y, true,
    );

    // Run batch kernel with same parameters
    let batch_pipeline = m.make_pipeline("pinning_search_batch");
    let batch_hits = m.search_pinning_batch_raw(
        &batch_pipeline,
        &midstate, &suffix, 5000, 4, 8, 0xFFFFFFFE, 1, 1024,
        &neg_r_inv, &u2r_x, &u2r_y, true, 256,
    );

    // Both should find the same hits (same candidates, same pipeline)
    let mut orig_lts: Vec<u32> = original_hits.iter().map(|h| h.locktime).collect();
    let mut batch_lts: Vec<u32> = batch_hits.iter().map(|h| h.locktime).collect();
    orig_lts.sort();
    batch_lts.sort();

    assert_eq!(
        orig_lts, batch_lts,
        "Batch kernel must find same hits as original.\nOriginal: {orig_lts:?}\nBatch: {batch_lts:?}"
    );
    // Sanity: should have found some easy-mode hits in 1024 candidates (~64 expected)
    assert!(!orig_lts.is_empty(), "should find at least one easy-mode hit in 1024 candidates");
}

/// Build a minimal test transaction and QSB script for GPU parameter tests.
fn build_test_tx_and_script() -> (tx::Transaction, Vec<u8>, hors::NonceSig) {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let config = script::QsbConfig::test();
    let pin_nonce = hors::NonceSig::derive("gpu_params_test_pin");
    let round1_nonce = hors::NonceSig::derive("gpu_params_test_r1");
    let round2_nonce = hors::NonceSig::derive("gpu_params_test_r2");

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let hors1 = hors::HorsKeys::generate(config.n, &mut rng);
    let hors2 = hors::HorsKeys::generate(config.n, &mut rng);
    let dummy1 = hors::generate_dummy_sigs(config.n, 0);
    let dummy2 = hors::generate_dummy_sigs(config.n, 1);

    let full_script = script::build_full_script(
        config,
        &pin_nonce.der_encoded,
        &round1_nonce.der_encoded,
        &round2_nonce.der_encoded,
        &[hors1.commitments, hors2.commitments],
        &[dummy1, dummy2],
    );

    let mut tx = tx::Transaction::new(2, 0);
    tx.add_input(tx::TxIn {
        txid: [0xAA; 32],
        vout: 0,
        script_sig: Vec::new(),
        sequence: 0xFFFFFFFE,
    });
    tx.add_output(tx::TxOut {
        value: 50_000,
        script_pubkey: vec![0x76, 0xa9, 0x14, /* 20 zero bytes */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x88, 0xac],
    });

    (tx, full_script, pin_nonce)
}

#[test]
fn gpu_search_params_ec_recovery_matches_cpu() {
    let m = miner();
    let (tx, full_script, pin_nonce) = build_test_tx_and_script();

    let params = GpuSearchParams::from_pinning_search(
        pin_nonce.parsed(),
        &tx,
        &full_script,
        &pin_nonce.der_encoded,
        0,
    );

    // Verify r * neg_r_inv = -1 mod N (scalar inversion correctness)
    {
        use secp256k1::{Scalar, SecretKey};
        let r_sk = SecretKey::from_byte_array(pin_nonce.parsed().r).expect("valid");
        let neg_r_inv_scalar = Scalar::from_be_bytes(params.neg_r_inv).expect("valid");
        let product = r_sk.mul_tweak(&neg_r_inv_scalar).expect("valid");
        let mut n_minus_1 = ecdsa_recovery::SECP256K1_N;
        n_minus_1[31] -= 1;
        assert_eq!(
            hex(&product.secret_bytes()), hex(&n_minus_1),
            "r * neg_r_inv should equal N-1 (i.e., -1 mod N)"
        );
    }

    // Verify GPU scalar_mul matches CPU for large values
    let pin_script_code = script::find_and_delete(&full_script, &pin_nonce.der_encoded);
    let test_locktime: u32 = 42;

    let mut test_tx = tx.clone();
    test_tx.inputs[0].sequence = 0xFFFFFFFE;
    test_tx.locktime = test_locktime;
    let cpu_digest = test_tx
        .legacy_sighash(0, &pin_script_code, pin_nonce.parsed().sighash_type)
        .expect("valid sighash");

    {
        use secp256k1::{Scalar, SecretKey};
        let nri_sk = SecretKey::from_byte_array(params.neg_r_inv).expect("valid");
        let z_scalar = Scalar::from_be_bytes(cpu_digest).expect("valid");
        let cpu_u1 = nri_sk.mul_tweak(&z_scalar).expect("valid").secret_bytes();
        let gpu_u1 = m.run_simple_kernel_2in_1out("test_scalar_mul", &params.neg_r_inv, &cpu_digest, 32);
        assert_eq!(hex(&cpu_u1), hex(&gpu_u1), "GPU scalar_mul must match CPU");
    }

    // Run GPU EC recovery with the CPU-computed digest and precomputed params
    let (gpu_pubkey, gpu_h160) = m.test_ec_recovery(
        &cpu_digest,
        &params.neg_r_inv,
        &params.u2r_x,
        &params.u2r_y,
    );

    // CPU recovery via secp256k1 library
    let (cpu_key, _recid) = ecdsa_recovery::recover_first_pubkey(pin_nonce.parsed(), cpu_digest)
        .expect("recovery should succeed");
    let cpu_pubkey = cpu_key.serialize();
    let cpu_h160 = hash::hash160(&cpu_pubkey);

    assert_eq!(
        hex(&gpu_pubkey), hex(&cpu_pubkey),
        "GPU EC recovery pubkey must match CPU"
    );
    assert_eq!(
        hex(&gpu_h160), hex(&cpu_h160),
        "GPU HASH160 must match CPU"
    );
}

#[test]
fn gpu_search_params_midstate_produces_correct_sighash() {
    let (tx, full_script, pin_nonce) = build_test_tx_and_script();

    let params = GpuSearchParams::from_pinning_search(
        pin_nonce.parsed(),
        &tx,
        &full_script,
        &pin_nonce.der_encoded,
        0,
    );

    let pin_script_code = script::find_and_delete(&full_script, &pin_nonce.der_encoded);
    let test_locktime: u32 = 100;
    let test_sequence: u32 = 0xFFFFFFFE;

    let mut test_tx = tx.clone();
    test_tx.inputs[0].sequence = test_sequence;
    test_tx.locktime = test_locktime;
    let preimage = test_tx
        .legacy_sighash_preimage(0, &pin_script_code, pin_nonce.parsed().sighash_type)
        .expect("valid preimage");

    assert_eq!(preimage.len(), params.total_preimage_len as usize);
    let midstate_boundary = preimage.len() - params.suffix.len();

    // Verify sequence and locktime offsets point to the right bytes
    let so = params.seq_offset as usize;
    let lo = params.lt_offset as usize;
    assert_eq!(
        &preimage[midstate_boundary + so..midstate_boundary + so + 4],
        &test_sequence.to_le_bytes(),
        "sequence offset must be correct"
    );
    assert_eq!(
        &preimage[midstate_boundary + lo..midstate_boundary + lo + 4],
        &test_locktime.to_le_bytes(),
        "locktime offset must be correct"
    );

    // Verify midstate matches sha256_midstate of the prefix
    let recomputed_midstate = hash::sha256_midstate(&preimage[..midstate_boundary]);
    assert_eq!(params.midstate, recomputed_midstate);
}

#[test]
fn gpu_pinning_hits_verified_by_cpu() {
    let m = miner();
    let (tx, full_script, pin_nonce) = build_test_tx_and_script();

    let params = GpuSearchParams::from_pinning_search(
        pin_nonce.parsed(),
        &tx,
        &full_script,
        &pin_nonce.der_encoded,
        0,
    );

    let pin_script_code = script::find_and_delete(&full_script, &pin_nonce.der_encoded);
    let start_lt: u32 = 1;
    let batch_size: u32 = 4096;
    let test_sequence: u32 = 0xFFFFFFFE;

    // Run GPU search with real params in easy mode
    let gpu_hits = m.search_pinning_batch(
        &params.midstate,
        &params.suffix,
        params.total_preimage_len,
        params.seq_offset,
        params.lt_offset,
        test_sequence,
        start_lt,
        batch_size,
        &params.neg_r_inv,
        &params.u2r_x,
        &params.u2r_y,
        true, // easy mode
    );

    println!("GPU found {} easy-mode hits in {} candidates", gpu_hits.len(), batch_size);
    assert!(!gpu_hits.is_empty(), "should find at least one easy-mode hit in 4096 candidates");

    // Verify every GPU hit against CPU
    let mut cpu_verified = 0;
    for hit in &gpu_hits {
        let mut verify_tx = tx.clone();
        verify_tx.inputs[0].sequence = test_sequence;
        verify_tx.locktime = hit.locktime;

        let cpu_digest = verify_tx
            .legacy_sighash(0, &pin_script_code, pin_nonce.parsed().sighash_type)
            .expect("valid sighash");

        let cpu_hit = puzzle::evaluate_puzzle(
            pin_nonce.parsed(),
            cpu_digest,
            puzzle::SearchMode::EasyTest,
        );

        assert!(
            cpu_hit.is_some(),
            "GPU hit at locktime={} must also be a CPU hit",
            hit.locktime
        );
        cpu_verified += 1;
    }

    println!("{cpu_verified}/{} GPU hits verified by CPU", gpu_hits.len());

    // Also check for CPU hits that the GPU missed (false negatives)
    let mut cpu_hits = Vec::new();
    for lt in start_lt..start_lt + batch_size {
        let mut verify_tx = tx.clone();
        verify_tx.inputs[0].sequence = test_sequence;
        verify_tx.locktime = lt;

        let cpu_digest = verify_tx
            .legacy_sighash(0, &pin_script_code, pin_nonce.parsed().sighash_type)
            .expect("valid sighash");

        if puzzle::evaluate_puzzle(
            pin_nonce.parsed(),
            cpu_digest,
            puzzle::SearchMode::EasyTest,
        ).is_some() {
            cpu_hits.push(lt);
        }
    }

    let mut gpu_lts: Vec<u32> = gpu_hits.iter().map(|h| h.locktime).collect();
    gpu_lts.sort();

    assert_eq!(
        gpu_lts, cpu_hits,
        "GPU and CPU must find identical hit sets.\nGPU: {gpu_lts:?}\nCPU: {cpu_hits:?}"
    );
    println!("Perfect parity: {} hits match between GPU and CPU", cpu_hits.len());
}

#[test]
fn gpu_cpu_search_parity() {
    let m = miner();
    let (tx, full_script, pin_nonce) = build_test_tx_and_script();

    let params = GpuSearchParams::from_pinning_search(
        pin_nonce.parsed(),
        &tx,
        &full_script,
        &pin_nonce.der_encoded,
        0,
    );

    let pin_script_code = script::find_and_delete(&full_script, &pin_nonce.der_encoded);
    let sequence = 0xFFFFFFFE_u32;
    let start_lt = 1_u32;
    let lt_count = 8192_u32;

    // GPU search
    let gpu_hits = m.search_pinning_batch(
        &params.midstate,
        &params.suffix,
        params.total_preimage_len,
        params.seq_offset,
        params.lt_offset,
        sequence,
        start_lt,
        lt_count,
        &params.neg_r_inv,
        &params.u2r_x,
        &params.u2r_y,
        true,
    );
    let mut gpu_lts: Vec<u32> = gpu_hits.iter().map(|h| h.locktime).collect();
    gpu_lts.sort();

    // CPU search using search::search_pinning with the same parameters
    let cpu_params = search::PinningSearchParams {
        tx: &tx,
        full_script: &full_script,
        pin_script_code: &pin_script_code,
        sig_nonce: pin_nonce.parsed(),
        sig_nonce_bytes: &pin_nonce.der_encoded,
        search_space: search::PinningSearchSpace {
            sequence_start: sequence,
            sequence_count: 1,
            locktime_start: start_lt,
            locktime_count: lt_count,
        },
        mode: puzzle::SearchMode::EasyTest,
        input_index: 0,
        tx_modifier: None,
    };

    // Collect ALL CPU hits (not just the first)
    let mut cpu_lts: Vec<u32> = Vec::new();
    for lt in start_lt..start_lt + lt_count {
        let mut verify_tx = tx.clone();
        verify_tx.inputs[0].sequence = sequence;
        verify_tx.locktime = lt;
        let digest = verify_tx
            .legacy_sighash(0, &pin_script_code, pin_nonce.parsed().sighash_type)
            .expect("valid");
        if puzzle::evaluate_puzzle(pin_nonce.parsed(), digest, puzzle::SearchMode::EasyTest).is_some() {
            cpu_lts.push(lt);
        }
    }

    // Verify the CPU search function finds a hit (sanity check)
    let first_cpu_hit = search::search_pinning(cpu_params);
    assert!(first_cpu_hit.is_some(), "CPU search should find at least one hit");
    let first_lt = first_cpu_hit.unwrap().locktime;
    assert!(cpu_lts.contains(&first_lt), "CPU search result should be in our exhaustive set");

    assert_eq!(
        gpu_lts, cpu_lts,
        "GPU and CPU must produce identical hit sets over {} candidates.\nGPU found {} hits, CPU found {} hits",
        lt_count, gpu_lts.len(), cpu_lts.len()
    );

    println!(
        "CPU↔GPU parity verified: {}/{} hits match over {} candidates",
        gpu_lts.len(), cpu_lts.len(), lt_count
    );
}

#[test]
fn gpu_batched_kernel_matches_original() {
    let m = miner();
    let (tx, full_script, pin_nonce) = build_test_tx_and_script();

    let params = GpuSearchParams::from_pinning_search(
        pin_nonce.parsed(),
        &tx,
        &full_script,
        &pin_nonce.der_encoded,
        0,
    );

    let start_lt = 1_u32;
    let batch_size = 8192_u32; // must be multiple of BATCH_N
    let sequence = 0xFFFFFFFE_u32;

    // Original kernel
    let original_hits = m.search_pinning_batch(
        &params.midstate,
        &params.suffix,
        params.total_preimage_len,
        params.seq_offset,
        params.lt_offset,
        sequence,
        start_lt,
        batch_size,
        &params.neg_r_inv,
        &params.u2r_x,
        &params.u2r_y,
        true,
    );

    // Batched kernel (per-thread Montgomery inversion)
    let batched_hits = m.search_pinning_batched(
        &params.midstate,
        &params.suffix,
        params.total_preimage_len,
        params.seq_offset,
        params.lt_offset,
        sequence,
        start_lt,
        batch_size,
        &params.neg_r_inv,
        &params.u2r_x,
        &params.u2r_y,
        true,
    );

    let mut orig_lts: Vec<u32> = original_hits.iter().map(|h| h.locktime).collect();
    let mut batch_lts: Vec<u32> = batched_hits.iter().map(|h| h.locktime).collect();
    orig_lts.sort();
    batch_lts.sort();

    assert_eq!(
        orig_lts, batch_lts,
        "Batched kernel must find identical hits.\nOriginal: {} hits\nBatched: {} hits",
        orig_lts.len(), batch_lts.len()
    );
    println!(
        "Batched kernel matches original: {}/{} hits over {} candidates",
        batch_lts.len(), orig_lts.len(), batch_size
    );
}
