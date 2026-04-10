use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn bench_hash(c: &mut Criterion) {
    let data = [0x42u8; 33]; // compressed pubkey size

    c.bench_function("sha256 (33 bytes)", |b| {
        b.iter(|| hash::sha256(black_box(&data)))
    });

    c.bench_function("sha256d (33 bytes)", |b| {
        b.iter(|| hash::sha256d(black_box(&data)))
    });

    c.bench_function("ripemd160 (32 bytes)", |b| {
        let sha = hash::sha256(&data);
        b.iter(|| hash::ripemd160(black_box(&sha)))
    });

    c.bench_function("hash160 (33 bytes)", |b| {
        b.iter(|| hash::hash160(black_box(&data)))
    });
}

fn bench_der(c: &mut Criterion) {
    // Valid 9-byte minimal DER sig
    let valid_sig = [0x30, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01, 0x01];
    // Random 20-byte input (typical RIPEMD-160 output)
    let random_20 = [0x42u8; 20];

    c.bench_function("is_valid_der_sig (9-byte valid)", |b| {
        b.iter(|| der::is_valid_der_sig(black_box(&valid_sig)))
    });

    c.bench_function("is_valid_der_sig (20-byte random)", |b| {
        b.iter(|| der::is_valid_der_sig(black_box(&random_20)))
    });

    c.bench_function("easy_der_predicate (20 bytes)", |b| {
        b.iter(|| der::easy_der_predicate(black_box(&random_20)))
    });

    let mut r = [0u8; 32];
    r[0] = 0x7F;
    r[31] = 0x01;
    let mut s = [0u8; 32];
    s[31] = 0x42;
    c.bench_function("encode_der_sig", |b| {
        b.iter(|| der::encode_der_sig(black_box(&r), black_box(&s), 0x01))
    });

    let encoded = der::encode_der_sig(&r, &s, 0x01);
    c.bench_function("parse_der_sig", |b| {
        b.iter(|| der::parse_der_sig(black_box(&encoded)))
    });
}

fn bench_ecdsa_recovery(c: &mut Criterion) {
    let r = ecdsa_recovery::derive_valid_xcoord("bench_r");
    let s = ecdsa_recovery::derive_valid_scalar("bench_s");
    let encoded = der::encode_der_sig(&r, &s, 0x01);
    let parsed = der::parse_der_sig(&encoded).unwrap();
    let digest = [0x42u8; 32];

    c.bench_function("recover_first_pubkey", |b| {
        b.iter(|| ecdsa_recovery::recover_first_pubkey(black_box(&parsed), black_box(digest)))
    });
}

fn bench_puzzle(c: &mut Criterion) {
    let r = ecdsa_recovery::derive_valid_xcoord("bench_r");
    let s = ecdsa_recovery::derive_valid_scalar("bench_s");
    let encoded = der::encode_der_sig(&r, &s, 0x01);
    let parsed = der::parse_der_sig(&encoded).unwrap();
    let digest = [0x42u8; 32];

    c.bench_function("evaluate_puzzle (production)", |b| {
        b.iter(|| {
            puzzle::evaluate_puzzle(
                black_box(&parsed),
                black_box(digest),
                puzzle::SearchMode::Production,
            )
        })
    });

    c.bench_function("evaluate_puzzle (easy test)", |b| {
        b.iter(|| {
            puzzle::evaluate_puzzle(
                black_box(&parsed),
                black_box(digest),
                puzzle::SearchMode::EasyTest,
            )
        })
    });
}

fn bench_subset(c: &mut Criterion) {
    c.bench_function("CombinationIter C(20,2) exhaust", |b| {
        b.iter(|| {
            let count = subset::CombinationIter::new(20, 2).count();
            black_box(count);
        })
    });

    c.bench_function("nth_combination C(150,9)", |b| {
        b.iter(|| subset::nth_combination(150, 9, black_box(41_000_000_000)))
    });

    c.bench_function("combination_index C(150,9)", |b| {
        let combo = vec![10, 25, 40, 55, 70, 85, 100, 120, 140];
        b.iter(|| subset::combination_index(black_box(&combo), 150))
    });

    c.bench_function("binomial_coefficient C(150,9)", |b| {
        b.iter(|| subset::binomial_coefficient(black_box(150), black_box(9)))
    });
}

fn bench_script(c: &mut Criterion) {
    // Build a realistic script for FindAndDelete benchmarks
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let config = script::QsbConfig::test();
    let hors0 = hors::HorsKeys::generate(config.n, &mut rng);
    let dummy0 = hors::generate_dummy_sigs(config.n, 0);
    let nonce = hors::NonceSig::derive("bench_round");
    let round_script = script::build_round_script(
        config.n,
        config.t1_signed,
        config.t1_bonus,
        &nonce.der_encoded,
        &hors0.commitments,
        &dummy0,
    );

    c.bench_function("find_and_delete (test script, 1 sig)", |b| {
        b.iter(|| script::find_and_delete(black_box(&round_script), &dummy0[0]))
    });
}

fn bench_sighash(c: &mut Criterion) {
    let mut t = tx::Transaction::new(1, 500_000);
    t.add_input(tx::TxIn {
        txid: [0xAA; 32],
        vout: 0,
        script_sig: Vec::new(),
        sequence: 0xffff_fffe,
    });
    t.add_input(tx::TxIn {
        txid: [0xBB; 32],
        vout: 0,
        script_sig: Vec::new(),
        sequence: 0xffff_fffe,
    });
    t.add_output(tx::TxOut {
        value: 45_000,
        script_pubkey: vec![0x76, 0xa9, 0x14, 0x00, 0x00, 0x00],
    });

    let script_code = vec![0x51; 100]; // 100-byte script code

    c.bench_function("legacy_sighash (100-byte script)", |b| {
        b.iter(|| {
            t.legacy_sighash(black_box(1), black_box(&script_code), tx::SIGHASH_ALL)
        })
    });

    // Realistic: 5KB script code (typical for a Binohash round after FindAndDelete)
    let big_script = vec![0x51; 5000];
    c.bench_function("legacy_sighash (5KB script)", |b| {
        b.iter(|| {
            t.legacy_sighash(black_box(1), black_box(&big_script), tx::SIGHASH_ALL)
        })
    });
}

criterion_group!(
    benches,
    bench_hash,
    bench_der,
    bench_ecdsa_recovery,
    bench_puzzle,
    bench_subset,
    bench_script,
    bench_sighash,
);
criterion_main!(benches);
