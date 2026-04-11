use criterion::{Criterion, black_box, criterion_group, criterion_main};
use metal_gpu::MetalMiner;
use std::sync::LazyLock;

static MINER: LazyLock<MetalMiner> = LazyLock::new(|| {
    let cache = std::env::temp_dir().join("binohash_bench/gtable.bin");
    std::fs::create_dir_all(cache.parent().unwrap()).ok();
    MetalMiner::new(Some(&cache)).expect("Metal device")
});

fn bench_gpu_sha256(c: &mut Criterion) {
    let m = &*MINER;
    let input = [0x42u8; 32];
    c.bench_function("GPU sha256 (32 bytes, single thread)", |b| {
        b.iter(|| m.test_sha256(black_box(&input)))
    });
}

fn bench_gpu_hash160(c: &mut Criterion) {
    let m = &*MINER;
    let mut input = [0u8; 33];
    input[0] = 0x02;
    input[1] = 0x79;
    c.bench_function("GPU hash160 (33 bytes, single thread)", |b| {
        b.iter(|| m.test_hash160(black_box(&input)))
    });
}

fn bench_gpu_field_inv(c: &mut Criterion) {
    let m = &*MINER;
    let mut a = [0u8; 32];
    a[31] = 42;
    c.bench_function("GPU field_inv (single thread)", |b| {
        b.iter(|| m.test_field_inv(black_box(&a)))
    });
}

fn bench_gpu_ec_mul(c: &mut Criterion) {
    let m = &*MINER;
    let scalar = ecdsa_recovery::derive_valid_scalar("bench_gpu_scalar");
    c.bench_function("GPU ec_mul via GTable (single thread)", |b| {
        b.iter(|| m.test_ec_mul(black_box(&scalar)))
    });
}

fn bench_gpu_pinning_batch(c: &mut Criterion) {
    let m = &*MINER;

    // Set up a realistic pinning search scenario
    let nonce = hors::NonceSig::derive("bench_pin");
    let parsed = nonce.parsed();

    // Precompute neg_r_inv and u2r on CPU (these are constants per nonce sig)
    // For the benchmark we just need plausible values — the kernel will run
    // regardless of whether a DER hit is found.
    let neg_r_inv = parsed.r; // Not mathematically correct, but valid bytes for benchmarking
    let u2r_x = parsed.r;
    let u2r_y = parsed.s;

    // Minimal suffix (just enough for the kernel to run)
    let suffix = vec![0u8; 32];
    let midstate = [0u32; 8]; // dummy midstate

    for batch_size in [1024u32, 65536, 262144] {
        c.bench_function(&format!("GPU pinning batch ({batch_size} candidates)"), |b| {
            b.iter(|| {
                m.search_pinning_batch(
                    black_box(&midstate),
                    black_box(&suffix),
                    5000, // total_preimage_len
                    4,    // seq_offset
                    8,    // lt_offset
                    0xFFFFFFFE,
                    1,    // start_lt
                    batch_size,
                    black_box(&neg_r_inv),
                    black_box(&u2r_x),
                    black_box(&u2r_y),
                    true, // easy_mode
                )
            })
        });
    }
}

fn bench_cpu_puzzle_comparison(c: &mut Criterion) {
    // CPU baseline: evaluate_puzzle (the per-candidate bottleneck)
    let r = ecdsa_recovery::derive_valid_xcoord("bench_r");
    let s = ecdsa_recovery::derive_valid_scalar("bench_s");
    let encoded = der::encode_der_sig(&r, &s, 0x01);
    let parsed = der::parse_der_sig(&encoded).unwrap();
    let digest = [0x42u8; 32];

    c.bench_function("CPU evaluate_puzzle (single candidate)", |b| {
        b.iter(|| {
            puzzle::evaluate_puzzle(
                black_box(&parsed),
                black_box(digest),
                puzzle::SearchMode::Production,
            )
        })
    });
}

criterion_group!(
    benches,
    bench_gpu_sha256,
    bench_gpu_hash160,
    bench_gpu_field_inv,
    bench_gpu_ec_mul,
    bench_gpu_pinning_batch,
    bench_cpu_puzzle_comparison,
);
criterion_main!(benches);
