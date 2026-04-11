use metal_gpu::{GpuSearchParams, MetalMiner};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::{Duration, Instant};

const BATCH_SIZE: u32 = 262_144;
const REPORT_INTERVAL: Duration = Duration::from_secs(30);
const RUN_DURATION: Duration = Duration::from_secs(3600); // 1 hour

fn main() {
    println!("=== GPU Full Pipeline Throughput Test ===");
    println!("Batch size: {BATCH_SIZE}");
    println!("Duration:   {} minutes", RUN_DURATION.as_secs() / 60);
    println!();

    // Build real search params
    println!("Building GpuSearchParams from real NonceSig + Transaction...");
    let config = script::QsbConfig::test();
    let pin_nonce = hors::NonceSig::derive("throughput_pin");
    let round1_nonce = hors::NonceSig::derive("throughput_r1");
    let round2_nonce = hors::NonceSig::derive("throughput_r2");

    let mut rng = ChaCha8Rng::seed_from_u64(777);
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
        txid: [0xCC; 32],
        vout: 0,
        script_sig: Vec::new(),
        sequence: 0xFFFFFFFE,
    });
    tx.add_output(tx::TxOut {
        value: 50_000,
        script_pubkey: vec![
            0x76, 0xa9, 0x14,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0x88, 0xac,
        ],
    });

    let params = GpuSearchParams::from_pinning_search(
        pin_nonce.parsed(),
        &tx,
        &full_script,
        &pin_nonce.der_encoded,
        0,
    );

    // Initialize Metal
    println!("Initializing Metal GPU...");
    let cache = std::env::temp_dir().join("binohash_throughput/gtable.bin");
    std::fs::create_dir_all(cache.parent().unwrap()).ok();
    let miner = MetalMiner::new(Some(&cache)).expect("Metal device");
    println!("Device: {}", miner.device_name());
    println!("Preimage: {} bytes, suffix: {} bytes", params.total_preimage_len, params.suffix.len());
    println!();

    // Run continuous batches
    let start = Instant::now();
    let mut last_report = start;
    let mut total_candidates: u64 = 0;
    let mut total_hits: u64 = 0;
    let mut batch_count: u64 = 0;
    let mut current_lt: u32 = 1;
    let sequence = 0xFFFFFFFE_u32;

    println!("Starting sustained throughput test...");
    println!("{:<12} {:>14} {:>10} {:>12} {:>10}", "Elapsed", "Candidates", "Hits", "Cand/sec", "Batch ms");
    println!("{}", "-".repeat(62));

    loop {
        let elapsed = start.elapsed();
        if elapsed >= RUN_DURATION {
            break;
        }

        let batch_start = Instant::now();
        let hits = miner.search_pinning_batch(
            &params.midstate,
            &params.suffix,
            params.total_preimage_len,
            params.seq_offset,
            params.lt_offset,
            sequence,
            current_lt,
            BATCH_SIZE,
            &params.neg_r_inv,
            &params.u2r_x,
            &params.u2r_y,
            true, // easy mode
        );
        let batch_ms = batch_start.elapsed().as_secs_f64() * 1000.0;

        total_candidates += BATCH_SIZE as u64;
        total_hits += hits.len() as u64;
        batch_count += 1;
        current_lt = current_lt.wrapping_add(BATCH_SIZE);

        // Report periodically
        if last_report.elapsed() >= REPORT_INTERVAL {
            let elapsed_secs = elapsed.as_secs_f64();
            let cand_per_sec = total_candidates as f64 / elapsed_secs;
            println!(
                "{:<12} {:>14} {:>10} {:>12.0} {:>9.1}ms",
                format!("{}m {:02}s", elapsed.as_secs() / 60, elapsed.as_secs() % 60),
                format_count(total_candidates),
                format_count(total_hits),
                cand_per_sec,
                batch_ms,
            );
            last_report = Instant::now();
        }
    }

    // Final report
    let total_secs = start.elapsed().as_secs_f64();
    let cand_per_sec = total_candidates as f64 / total_secs;
    let hit_rate = total_hits as f64 / total_candidates as f64;
    println!("{}", "=".repeat(62));
    println!("FINAL RESULTS");
    println!("  Duration:       {:.1} minutes", total_secs / 60.0);
    println!("  Batches:        {}", format_count(batch_count));
    println!("  Candidates:     {}", format_count(total_candidates));
    println!("  Hits:           {} ({:.2}% hit rate)", format_count(total_hits), hit_rate * 100.0);
    println!("  Throughput:     {:.0} candidates/sec", cand_per_sec);
    println!("  Avg batch:      {:.1} ms", total_secs * 1000.0 / batch_count as f64);
    println!("  µs/candidate:   {:.3}", total_secs * 1_000_000.0 / total_candidates as f64);
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{n}")
    }
}
