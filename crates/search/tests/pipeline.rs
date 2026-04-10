use hors::{HorsKeys, NonceSig, generate_dummy_sigs};
use puzzle::SearchMode;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use search::{
    DigestSearchParams, PinningSearchParams, PinningSearchSpace, search_digest, search_pinning,
};
use script::{QsbConfig, build_full_script, find_and_delete};
use tx::{Transaction, TxIn, TxOut};

struct TestFixture {
    config: QsbConfig,
    full_script: Vec<u8>,
    pin_sig: NonceSig,
    round_sigs: [NonceSig; 2],
    dummy_sigs: [Vec<[u8; 9]>; 2],
    hors_keys: [HorsKeys; 2],
    template_tx: Transaction,
}

impl TestFixture {
    fn new() -> Self {
        let config = QsbConfig::test();
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let hors0 = HorsKeys::generate(config.n, &mut rng);
        let hors1 = HorsKeys::generate(config.n, &mut rng);
        let dummy_sigs0 = generate_dummy_sigs(config.n, 0);
        let dummy_sigs1 = generate_dummy_sigs(config.n, 1);

        let pin_sig = NonceSig::derive("qsb_pin");
        let round1_sig = NonceSig::derive("qsb_round1");
        let round2_sig = NonceSig::derive("qsb_round2");

        let full_script = build_full_script(
            config,
            &pin_sig.der_encoded,
            &round1_sig.der_encoded,
            &round2_sig.der_encoded,
            &[hors0.commitments.clone(), hors1.commitments.clone()],
            &[dummy_sigs0.clone(), dummy_sigs1.clone()],
        );

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
            script_pubkey: vec![
                0x76, 0xa9, 0x14,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x88, 0xac,
            ],
        });

        TestFixture {
            config,
            full_script,
            pin_sig,
            round_sigs: [round1_sig, round2_sig],
            dummy_sigs: [dummy_sigs0, dummy_sigs1],
            hors_keys: [hors0, hors1],
            template_tx,
        }
    }

    fn t_signed(&self, round: usize) -> usize {
        if round == 0 { self.config.t1_signed } else { self.config.t2_signed }
    }

    fn t_bonus(&self, round: usize) -> usize {
        if round == 0 { self.config.t1_bonus } else { self.config.t2_bonus }
    }
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
        sig_nonce_bytes: &f.pin_sig.der_encoded,
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
        sig_nonce_bytes: &f.pin_sig.der_encoded,
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
        sig_nonce_bytes: &f.round_sigs[0].der_encoded,
        dummy_sigs: &f.dummy_sigs[0],
        hors_secrets: &f.hors_keys[0].secrets,
        n: f.config.n,
        t_signed: f.t_signed(0),
        t_bonus: f.t_bonus(0),
        mode: SearchMode::EasyTest,
        input_index: 1,
    });

    assert!(round_result.is_some(), "should find digest hit for round 1");
    let hit = round_result.unwrap();
    assert_eq!(hit.signed_indices.len(), f.t_signed(0));
    assert_eq!(hit.bonus_indices.len(), f.t_bonus(0));
    assert_eq!(hit.indices.len(), f.t_signed(0) + f.t_bonus(0));
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
        sig_nonce_bytes: &f.pin_sig.der_encoded,
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
            sig_nonce_bytes: &f.round_sigs[round].der_encoded,
            dummy_sigs: &f.dummy_sigs[round],
            hors_secrets: &f.hors_keys[round].secrets,
            n: f.config.n,
            t_signed: f.t_signed(round),
            t_bonus: f.t_bonus(round),
            mode: SearchMode::EasyTest,
            input_index: 1,
        });
        assert!(
            result.is_some(),
            "should find digest hit for round {}", round + 1
        );

        let hit = result.unwrap();
        // Verify HORS preimages match commitments
        for &idx in &hit.signed_indices {
            assert_eq!(
                hash::hash160(&f.hors_keys[round].secrets[idx]),
                f.hors_keys[round].commitments[idx],
                "HORS preimage mismatch at index {idx}"
            );
        }
    }
}
