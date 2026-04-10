use hors::{HorsKeys, NonceSig, generate_dummy_sigs};
use puzzle::SearchMode;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use search::{
    DigestSearchParams, PinningSearchParams, PinningSearchSpace, search_digest, search_pinning,
};
use script::find_and_delete;
use tx::{Transaction, TxIn, TxOut};

/// Build the test fixture used across all pipeline tests.
/// Mirrors the nktkt "test" config: n=10, t_signed=2, t_bonus=0.
struct TestFixture {
    n: usize,
    t_signed: usize,
    t_bonus: usize,
    full_script: Vec<u8>,
    pin_sig: NonceSig,
    round_sigs: [NonceSig; 2],
    dummy_sigs: [Vec<[u8; 9]>; 2],
    hors_keys: [HorsKeys; 2],
    template_tx: Transaction,
}

impl TestFixture {
    fn new() -> Self {
        let n = 20;
        let t_signed = 2;
        let t_bonus = 0;

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let hors0 = HorsKeys::generate(n, &mut rng);
        let hors1 = HorsKeys::generate(n, &mut rng);
        let dummy_sigs0 = generate_dummy_sigs(n, 0);
        let dummy_sigs1 = generate_dummy_sigs(n, 1);

        let pin_sig = NonceSig::derive("qsb_pin");
        let round1_sig = NonceSig::derive("qsb_round1");
        let round2_sig = NonceSig::derive("qsb_round2");

        // Build a minimal full script (pinning + 2 rounds of data)
        // For the test, we just concatenate the data that would be in the script:
        // HORS commitments, dummy sigs, OP_0, sig_nonce for each round.
        let full_script = build_test_script(
            n,
            t_signed,
            t_bonus,
            &pin_sig,
            &round1_sig,
            &round2_sig,
            &hors0,
            &hors1,
            &dummy_sigs0,
            &dummy_sigs1,
        );

        // Build template transaction with 2 inputs and 1 output
        let mut template_tx = Transaction::new(1, 0);
        template_tx.add_input(TxIn {
            txid: [0u8; 32],
            vout: 0,
            script_sig: Vec::new(),
            sequence: 0xffff_fffe,
        });
        template_tx.add_input(TxIn {
            txid: [1u8; 32],
            vout: 0,
            script_sig: Vec::new(),
            sequence: 0xffff_fffe,
        });
        template_tx.add_output(TxOut {
            value: 45_000,
            script_pubkey: vec![0x76, 0xa9, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                               0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                               0x00, 0x00, 0x00, 0x88, 0xac],
        });

        TestFixture {
            n,
            t_signed,
            t_bonus,
            full_script,
            pin_sig,
            round_sigs: [round1_sig, round2_sig],
            dummy_sigs: [dummy_sigs0, dummy_sigs1],
            hors_keys: [hors0, hors1],
            template_tx,
        }
    }
}

fn build_test_script(
    n: usize,
    t_signed: usize,
    t_bonus: usize,
    pin_sig: &NonceSig,
    round1_sig: &NonceSig,
    round2_sig: &NonceSig,
    hors0: &HorsKeys,
    hors1: &HorsKeys,
    dummy_sigs0: &[[u8; 9]],
    dummy_sigs1: &[[u8; 9]],
) -> Vec<u8> {
    use script::*;

    let mut s = Vec::new();

    // Pinning section (5 non-push opcodes)
    s.extend_from_slice(&push_data(&pin_sig.der_encoded));
    s.push(OP_OVER);
    s.push(OP_CHECKSIGVERIFY);
    s.push(OP_RIPEMD160);
    s.push(OP_SWAP);
    s.push(OP_CHECKSIGVERIFY);

    // Round 1
    build_round_script(&mut s, n, t_signed, t_bonus, round1_sig, hors0, dummy_sigs0);

    // Round 2
    build_round_script(&mut s, n, t_signed, t_bonus, round2_sig, hors1, dummy_sigs1);

    s
}

fn build_round_script(
    s: &mut Vec<u8>,
    n: usize,
    t_signed: usize,
    t_bonus: usize,
    sig_nonce: &NonceSig,
    hors: &HorsKeys,
    dummy_sigs: &[[u8; 9]],
) {
    use script::*;
    let t_total = t_signed + t_bonus;

    for commitment in hors.commitments.iter().rev() {
        s.extend_from_slice(&push_data(commitment));
    }
    for dummy in dummy_sigs.iter().rev() {
        s.extend_from_slice(&push_data(dummy));
    }
    s.push(OP_0);
    s.extend_from_slice(&push_data(&sig_nonce.der_encoded));

    for i in 0..t_signed {
        let idx_pos = 2 * n + 1 - i;
        let sanitize = n - i;
        let preimage_pos = 2 * n + 1 + t_total - 2 * i;
        s.extend_from_slice(&push_number(idx_pos as i64));
        s.push(OP_ROLL);
        s.extend_from_slice(&push_number(sanitize as i64));
        s.push(OP_MIN);
        s.push(OP_DUP);
        s.extend_from_slice(&push_number((n + 1) as i64));
        s.push(OP_ADD);
        s.push(OP_ROLL);
        s.extend_from_slice(&push_number(preimage_pos as i64));
        s.push(OP_ROLL);
        s.push(OP_HASH160);
        s.push(OP_EQUALVERIFY);
        s.push(OP_ROLL);
    }

    for i in 0..t_bonus {
        let j = t_signed + i;
        let idx_pos = 2 * n + 1 - j;
        let sanitize = n - j;
        s.extend_from_slice(&push_number(idx_pos as i64));
        s.push(OP_ROLL);
        s.extend_from_slice(&push_number(sanitize as i64));
        s.push(OP_MIN);
        s.push(OP_ROLL);
    }

    let puzzle_pos = 2 * n + 2;
    s.extend_from_slice(&push_number(puzzle_pos as i64));
    s.push(OP_ROLL);
    s.push(OP_DUP);
    s.push(OP_RIPEMD160);
    s.extend_from_slice(&push_number(puzzle_pos as i64));
    s.push(OP_ROLL);
    s.push(OP_CHECKSIGVERIFY);

    let m = t_total + 1;
    s.extend_from_slice(&push_number(m as i64));
    s.push(OP_2);
    s.push(OP_ROLL);

    let cms_roll_pos = 2 * n + 3;
    for _ in 0..t_total {
        s.extend_from_slice(&push_number(cms_roll_pos as i64));
        s.push(OP_ROLL);
    }

    s.extend_from_slice(&push_number(m as i64));
    s.push(OP_CHECKMULTISIG);
}

#[test]
fn pinning_search_finds_hit_in_easy_mode() {
    let f = TestFixture::new();
    let pin_script_code = find_and_delete(&f.full_script, &f.pin_sig.der_encoded);

    let result = search_pinning(PinningSearchParams {
        tx: &f.template_tx,
        full_script: &f.full_script,
        pin_script_code: &pin_script_code,
        sig_nonce: &f.pin_sig.parsed(),
        search_space: PinningSearchSpace {
            sequence_start: 0xffff_fffe,
            sequence_count: 1,
            locktime_start: 1,
            locktime_count: 10_000,
        },
        mode: SearchMode::EasyTest,
        input_index: 1,
    });

    assert!(result.is_some(), "should find a pinning hit in easy mode");
    let hit = result.unwrap();
    assert!(hit.locktime >= 1 && hit.locktime <= 10_000);
}

#[test]
fn digest_search_finds_hit_in_easy_mode() {
    let f = TestFixture::new();
    let pin_script_code = find_and_delete(&f.full_script, &f.pin_sig.der_encoded);

    // First find a pinning solution
    let pin_hit = search_pinning(PinningSearchParams {
        tx: &f.template_tx,
        full_script: &f.full_script,
        pin_script_code: &pin_script_code,
        sig_nonce: &f.pin_sig.parsed(),
        search_space: PinningSearchSpace {
            sequence_start: 0xffff_fffe,
            sequence_count: 1,
            locktime_start: 1,
            locktime_count: 10_000,
        },
        mode: SearchMode::EasyTest,
        input_index: 1,
    })
    .expect("pinning should succeed in easy mode");

    // Fix the transaction with the pinning solution
    let mut tx = f.template_tx.clone();
    tx.inputs[1].sequence = pin_hit.sequence;
    tx.locktime = pin_hit.locktime;

    // Search for digest round 1
    let round_result = search_digest(DigestSearchParams {
        tx: &tx,
        full_script: &f.full_script,
        sig_nonce: &f.round_sigs[0].parsed(),
        dummy_sigs: &f.dummy_sigs[0],
        hors_secrets: &f.hors_keys[0].secrets,
        n: f.n,
        t_signed: f.t_signed,
        t_bonus: f.t_bonus,
        mode: SearchMode::EasyTest,
        input_index: 1,
    });

    assert!(round_result.is_some(), "should find digest hit for round 1");
    let hit = round_result.unwrap();
    assert_eq!(hit.indices.len(), f.t_signed + f.t_bonus);
}

#[test]
fn full_pipeline_easy_mode() {
    let f = TestFixture::new();
    let pin_script_code = find_and_delete(&f.full_script, &f.pin_sig.der_encoded);

    let pin_hit = search_pinning(PinningSearchParams {
        tx: &f.template_tx,
        full_script: &f.full_script,
        pin_script_code: &pin_script_code,
        sig_nonce: &f.pin_sig.parsed(),
        search_space: PinningSearchSpace {
            sequence_start: 0xffff_fffe,
            sequence_count: 1,
            locktime_start: 1,
            locktime_count: 10_000,
        },
        mode: SearchMode::EasyTest,
        input_index: 1,
    })
    .expect("pinning should succeed");

    let mut tx = f.template_tx.clone();
    tx.inputs[1].sequence = pin_hit.sequence;
    tx.locktime = pin_hit.locktime;

    for round in 0..2 {
        let result = search_digest(DigestSearchParams {
            tx: &tx,
            full_script: &f.full_script,
            sig_nonce: &f.round_sigs[round].parsed(),
            dummy_sigs: &f.dummy_sigs[round],
            hors_secrets: &f.hors_keys[round].secrets,
            n: f.n,
            t_signed: f.t_signed,
            t_bonus: f.t_bonus,
            mode: SearchMode::EasyTest,
            input_index: 1,
        });
        assert!(
            result.is_some(),
            "should find digest hit for round {}", round + 1
        );

        let hit = result.unwrap();
        // Verify HORS preimages match commitments
        for &idx in &hit.indices[..f.t_signed] {
            assert_eq!(
                hash::hash160(&f.hors_keys[round].secrets[idx]),
                f.hors_keys[round].commitments[idx],
                "HORS preimage mismatch at index {idx}"
            );
        }
    }
}
