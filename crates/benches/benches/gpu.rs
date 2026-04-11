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

    // Batch inversion pinning kernel (262144 candidates, tg=256)
    let batch_pipeline = m.make_pipeline("pinning_search_batch");
    c.bench_function("GPU pinning batch-inv (262144 candidates)", |b| {
        b.iter(|| {
            // Reuse the same buffer setup as search_pinning_batch
            m.search_pinning_batch_raw(
                &batch_pipeline,
                &midstate,
                &suffix,
                5000, 4, 8, 0xFFFFFFFE, 1, 262144,
                &neg_r_inv, &u2r_x, &u2r_y, true,
                256, // explicit threadgroup size
            )
        })
    });
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

fn bench_field_throughput(c: &mut Criterion) {
    let m = &*MINER;
    let threads = 262144u32;
    let iters = 100u32;

    // field_mul throughput: threads × iterations multiplications
    c.bench_function(
        &format!("GPU field_mul throughput ({threads}t × {iters}i = {}M muls)",
                 threads as u64 * iters as u64 / 1_000_000),
        |b| b.iter(|| m.bench_field_op("bench_field_mul", threads, iters)),
    );

    // field_sqr throughput (currently same as field_mul)
    c.bench_function(
        &format!("GPU field_sqr throughput ({threads}t × {iters}i = {}M sqrs)",
                 threads as u64 * iters as u64 / 1_000_000),
        |b| b.iter(|| m.bench_field_op("bench_field_sqr", threads, iters)),
    );

    // field_inv throughput: each inv = 271 field_muls
    let inv_iters = 1u32;
    c.bench_function(
        &format!("GPU field_inv throughput ({threads}t × {inv_iters}i)"),
        |b| b.iter(|| m.bench_field_op("bench_field_inv_loop", threads, inv_iters)),
    );

    // Montgomery mul throughput (13-bit CIOS)
    c.bench_function(
        &format!("GPU monty13_mul throughput ({threads}t × {iters}i = {}M muls)",
                 threads as u64 * iters as u64 / 1_000_000),
        |b| b.iter(|| m.bench_field_op("bench_monty_mul", threads, iters)),
    );

    // Montgomery sqr throughput
    c.bench_function(
        &format!("GPU monty13_sqr throughput ({threads}t × {iters}i = {}M sqrs)",
                 threads as u64 * iters as u64 / 1_000_000),
        |b| b.iter(|| m.bench_field_op("bench_monty_sqr", threads, iters)),
    );

    // Reference msl-secp256k1 mont_mul_optimised (verbatim algorithm, p from buffer)
    c.bench_function(
        &format!("GPU ref_mont_mul throughput ({threads}t × {iters}i = {}M muls)",
                 threads as u64 * iters as u64 / 1_000_000),
        |b| b.iter(|| m.bench_field_op("bench_ref_mont_mul", threads, iters)),
    );
}

fn bench_ec_comparison(c: &mut Criterion) {
    let m = &*MINER;
    let iters = 16u32;

    // The geo (msl-secp256k1) EC kernel uses 20×13-bit limbs which exceeds
    // Apple Silicon's per-thread register limit — pipeline creation fails.
    // This confirms our 8×32-bit approach is better for GPU parallelism.
    let geo_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        m.make_pipeline("bench_geo_ec_add_16")
    }));
    match geo_result {
        Ok(pipeline) => {
            for threads in [1u32, 256, 4096] {
                c.bench_function(
                    &format!("GPU geo_ec jac_add ({threads}t x {iters}i)"),
                    |b| b.iter(|| m.run_pipeline(&pipeline, threads, iters)),
                );
            }
        }
        Err(_) => {
            eprintln!("NOTE: geo_ec pipeline creation failed (register limit exceeded)");
        }
    }

    let our_pipeline = m.make_pipeline("bench_our_ec_add_16");
    for threads in [1u32, 256, 65536] {
        c.bench_function(
            &format!("GPU our_ec mixed_add ({threads}t x {iters}i)"),
            |b| b.iter(|| m.run_pipeline(&our_pipeline, threads, iters)),
        );
    }
}

fn bench_batch_inversion(c: &mut Criterion) {
    let m = &*MINER;

    // Per-thread inversion: each of 65536 threads does 1 field_inv
    let per_thread = m.make_pipeline("bench_inv_per_thread");
    c.bench_function("GPU inv per-thread (65536t x 1i)", |b| {
        b.iter(|| m.run_pipeline(&per_thread, 65536, 1))
    });

    // Batch inversion v1 (sequential): 65536 threads in groups of 256
    let batch = m.make_pipeline("bench_inv_batch");

    // Batch inversion v2 (parallel tree): all threads participate
    let batch_tree = m.make_pipeline("bench_inv_batch_tree");

    // Need custom dispatch with explicit threadgroup size = 256
    c.bench_function("GPU inv batch (65536t x 1i, tg=256)", |b| {
        b.iter(|| {
            let seed: [u8; 32] = [0x42; 32];
            let seed_buf = m.device().new_buffer_with_data(
                seed.as_ptr() as *const _, 32,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let out_buf = m.device().new_buffer(
                32, metal::MTLResourceOptions::StorageModeShared,
            );
            let iters: u32 = 1;
            let iter_buf = m.device().new_buffer_with_data(
                &iters as *const u32 as *const _, 4,
                metal::MTLResourceOptions::StorageModeShared,
            );

            let cmd = m.queue().new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&batch);
            enc.set_buffer(0, Some(&seed_buf), 0);
            enc.set_buffer(1, Some(&out_buf), 0);
            enc.set_buffer(2, Some(&iter_buf), 0);
            // Explicit threadgroup size = 256 to match BATCH_SIZE
            enc.dispatch_threads(
                metal::MTLSize::new(65536, 1, 1),
                metal::MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        })
    });

    c.bench_function("GPU inv batch-tree (65536t x 1i, tg=256)", |b| {
        b.iter(|| {
            let seed: [u8; 32] = [0x42; 32];
            let seed_buf = m.device().new_buffer_with_data(
                seed.as_ptr() as *const _, 32,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let out_buf = m.device().new_buffer(
                32, metal::MTLResourceOptions::StorageModeShared,
            );
            let iters: u32 = 1;
            let iter_buf = m.device().new_buffer_with_data(
                &iters as *const u32 as *const _, 4,
                metal::MTLResourceOptions::StorageModeShared,
            );

            let cmd = m.queue().new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&batch_tree);
            enc.set_buffer(0, Some(&seed_buf), 0);
            enc.set_buffer(1, Some(&out_buf), 0);
            enc.set_buffer(2, Some(&iter_buf), 0);
            enc.dispatch_threads(
                metal::MTLSize::new(65536, 1, 1),
                metal::MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
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
    bench_field_throughput,
    bench_ec_comparison,
    bench_batch_inversion,
);
criterion_main!(benches);
