//! End-to-end pipeline throughput benchmark for the QSB scheme.
//!
//! Runs each of the three production phases (pinning, Round 1 digest,
//! Round 2 digest) for a fixed wall-clock duration and reports the
//! sustained throughput of each. Then projects the total cost of a
//! complete Config A run against the ~2^46 puzzle target per phase.

use metal_gpu::{GpuDigestSearchParams, GpuSearchParams, MetalMiner};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::{Duration, Instant};

const PHASE_DURATION: Duration = Duration::from_secs(20);
const PINNING_BATCH: u32 = 262_144;
const DIGEST_BATCH: u32 = 65_536;

fn main() {
    println!("=== QSB End-to-End GPU Pipeline Throughput ===");
    println!("Phase duration: {} seconds each", PHASE_DURATION.as_secs());
    println!();

    // --- Setup: Config A full scenario ---
    let config = script::QsbConfig::config_a();
    let pin_nonce = hors::NonceSig::derive("pipeline_pin");
    let round1_nonce = hors::NonceSig::derive("pipeline_r1");
    let round2_nonce = hors::NonceSig::derive("pipeline_r2");

    let mut rng = ChaCha8Rng::seed_from_u64(2026);
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
        &[dummy1.clone(), dummy2.clone()],
    );

    let mut tx = tx::Transaction::new(2, 0);
    tx.add_input(tx::TxIn {
        txid: [0xAB; 32],
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

    println!("Building search params...");
    let pin_params = GpuSearchParams::from_pinning_search(
        pin_nonce.parsed(), &tx, &full_script, &pin_nonce.der_encoded, 0,
    );

    let dummy1_vecs: Vec<Vec<u8>> = dummy1.iter().map(|d| d.to_vec()).collect();
    let r1_params = GpuDigestSearchParams::from_digest_search(
        round1_nonce.parsed(),
        &round1_nonce.der_encoded,
        &dummy1_vecs,
        &tx,
        &full_script,
        0,
        config.t1_total(),
    );

    let dummy2_vecs: Vec<Vec<u8>> = dummy2.iter().map(|d| d.to_vec()).collect();
    let r2_params = GpuDigestSearchParams::from_digest_search(
        round2_nonce.parsed(),
        &round2_nonce.der_encoded,
        &dummy2_vecs,
        &tx,
        &full_script,
        0,
        config.t2_total(),
    );

    println!("Initializing Metal...");
    let cache = std::env::temp_dir().join("binohash_pipeline/gtable.bin");
    std::fs::create_dir_all(cache.parent().unwrap()).ok();
    let miner = MetalMiner::new(Some(&cache)).expect("Metal device");
    println!("Device: {}", miner.device_name());
    println!("Script size: {} bytes", full_script.len());
    println!(
        "Pinning params: preimage={} bytes, suffix={} bytes",
        pin_params.total_preimage_len, pin_params.suffix.len()
    );
    println!(
        "Round 1 params: base_tail={} bytes, n={}, t={}",
        r1_params.base_tail.len(), config.n, config.t1_total()
    );
    println!(
        "Round 2 params: base_tail={} bytes, n={}, t={}",
        r2_params.base_tail.len(), config.n, config.t2_total()
    );
    println!();

    // --- Phase 1: Pinning (batched N=8) ---
    println!("─── Phase 1: Pinning (batched N=8 kernel) ───");
    let pinning_rate = run_pinning(&miner, &pin_params);
    println!();

    // --- Phase 2: Round 1 digest (nth kernel) ---
    println!("─── Phase 2: Round 1 digest (nth_combination kernel) ───");
    let r1_rate = run_digest(
        &miner, &r1_params, config.t1_total(), config.n, "Round 1",
    );
    println!();

    // --- Phase 3: Round 2 digest (nth kernel) ---
    println!("─── Phase 3: Round 2 digest (nth_combination kernel) ───");
    let r2_rate = run_digest(
        &miner, &r2_params, config.t2_total(), config.n, "Round 2",
    );
    println!();

    // --- Summary & Config A cost projection ---
    println!("══════════════════════════════════════════════════════");
    println!("                PRODUCTION PROJECTIONS");
    println!("══════════════════════════════════════════════════════");
    println!();
    println!("Phase throughputs (sustained):");
    println!("  Pinning:       {:>10.2} M candidates/sec", pinning_rate / 1e6);
    println!("  Round 1 digest:{:>10.2} M subsets/sec", r1_rate / 1e6);
    println!("  Round 2 digest:{:>10.2} M subsets/sec", r2_rate / 1e6);
    println!();

    // Config A target: 2^46 pinning candidates, 2^46.2 per digest round
    let pinning_target = 2f64.powi(46);
    let digest_target = 2f64.powf(46.2);

    let pin_hours = pinning_target / pinning_rate / 3600.0;
    let r1_hours = digest_target / r1_rate / 3600.0;
    let r2_hours = digest_target / r2_rate / 3600.0;
    let total_hours = pin_hours + r1_hours + r2_hours;

    println!("Config A search cost (single M4 Pro):");
    println!("  Pinning (2^46):       {:>8.2} hours", pin_hours);
    println!("  Round 1 (2^46.2):     {:>8.2} hours", r1_hours);
    println!("  Round 2 (2^46.2):     {:>8.2} hours", r2_hours);
    println!("  Total:                {:>8.2} hours ({:.1} days)", total_hours, total_hours / 24.0);
    println!();

    // Rough cloud cost estimate at $0.50/hr equivalent GPU
    let dollars = total_hours * 0.50;
    println!("Rough cloud cost estimate @ $0.50/hr: ${:.2}", dollars);
    println!("(Multi-GPU scales linearly — 10 GPUs ≈ {:.1} hours, same cost)", total_hours / 10.0);
}

fn run_pinning(miner: &MetalMiner, params: &GpuSearchParams) -> f64 {
    let start = Instant::now();
    let mut total_candidates: u64 = 0;
    let mut total_hits: u64 = 0;
    let mut current_lt: u32 = 1;

    loop {
        let elapsed = start.elapsed();
        if elapsed >= PHASE_DURATION {
            break;
        }
        let hits = miner.search_pinning_batched(
            &params.midstate,
            &params.suffix,
            params.total_preimage_len,
            params.seq_offset,
            params.lt_offset,
            0xFFFFFFFE,
            current_lt,
            PINNING_BATCH,
            &params.neg_r_inv,
            &params.u2r_x,
            &params.u2r_y,
            true,
        );
        total_candidates += PINNING_BATCH as u64;
        total_hits += hits.len() as u64;
        current_lt = current_lt.wrapping_add(PINNING_BATCH);
    }

    let elapsed_secs = start.elapsed().as_secs_f64();
    let rate = total_candidates as f64 / elapsed_secs;
    let hit_rate = total_hits as f64 / total_candidates as f64 * 100.0;
    println!(
        "  Candidates: {} | Hits: {} ({:.3}%) | Rate: {:.2} M/sec",
        format_count(total_candidates), format_count(total_hits), hit_rate, rate / 1e6,
    );
    rate
}

fn run_digest(
    miner: &MetalMiner,
    params: &GpuDigestSearchParams,
    t: usize,
    n: usize,
    label: &str,
) -> f64 {
    let start = Instant::now();
    let mut total_candidates: u64 = 0;
    let mut total_hits: u64 = 0;
    let mut start_index: u64 = 0;

    loop {
        let elapsed = start.elapsed();
        if elapsed >= PHASE_DURATION {
            break;
        }
        let hits = miner.search_digest_batch_nth(
            params,
            t as u32,
            n as u32,
            start_index,
            DIGEST_BATCH,
            true,
        );
        total_candidates += DIGEST_BATCH as u64;
        total_hits += hits.len() as u64;
        start_index += DIGEST_BATCH as u64;
    }

    let elapsed_secs = start.elapsed().as_secs_f64();
    let rate = total_candidates as f64 / elapsed_secs;
    let hit_rate = total_hits as f64 / total_candidates as f64 * 100.0;
    println!(
        "  {} | Candidates: {} | Hits: {} ({:.3}%) | Rate: {:.2} M/sec",
        label, format_count(total_candidates), format_count(total_hits), hit_rate, rate / 1e6,
    );
    rate
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
