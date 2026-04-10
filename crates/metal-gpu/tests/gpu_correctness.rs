use metal_gpu::MetalMiner;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn miner() -> MetalMiner {
    MetalMiner::new(None).expect("Metal device and GTable should be available")
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
