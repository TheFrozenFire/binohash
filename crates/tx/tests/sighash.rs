use tx::{Transaction, TxIn, TxOut, SIGHASH_ALL, SIGHASH_NONE, SIGHASH_SINGLE, SIGHASH_ANYONECANPAY};

#[test]
fn sighash_single_bug_returns_one() {
    let mut tx = Transaction::new(1, 0);
    tx.add_input(TxIn {
        txid: [0u8; 32],
        vout: 0,
        script_sig: Vec::new(),
        sequence: 0xffff_fffe,
    });
    tx.add_input(TxIn {
        txid: [1u8; 32],
        vout: 1,
        script_sig: Vec::new(),
        sequence: 0xffff_fffe,
    });
    tx.add_output(TxOut {
        value: 50_000,
        script_pubkey: vec![0x51],
    });

    // input_index=1 but only 1 output → SIGHASH_SINGLE bug
    let digest = tx
        .legacy_sighash(1, &[0x51], SIGHASH_SINGLE)
        .expect("sighash");
    assert_eq!(digest[31], 1);
    assert!(digest[..31].iter().all(|b| *b == 0));
}

#[test]
fn legacy_sighash_matches_bitcoin_crate() {
    use bitcoin::{
        ScriptBuf,
        consensus::deserialize,
        hashes::Hash,
        sighash::SighashCache,
    };

    let mut tx = Transaction::new(2, 0x1020_3040);
    tx.add_input(TxIn {
        txid: [0x11; 32],
        vout: 1,
        script_sig: vec![0x51, 0x21, 0x02],
        sequence: 0x1122_3344,
    });
    tx.add_input(TxIn {
        txid: [0x22; 32],
        vout: 3,
        script_sig: vec![0x6a, 0x02, 0xab, 0xcd],
        sequence: 0x5566_7788,
    });
    tx.add_output(TxOut {
        value: 42_000,
        script_pubkey: vec![0x76, 0xa9, 0x14, 0x01, 0x88, 0xac],
    });
    tx.add_output(TxOut {
        value: 7_000,
        script_pubkey: vec![0x51, 0x21, 0x03],
    });

    let script_code = vec![0x76, 0xa9, 0x14, 0xde, 0xad, 0xbe, 0xef, 0x88, 0xac];
    let btc_tx: bitcoin::Transaction = deserialize(&tx.serialize())
        .expect("our serialization should be valid for the bitcoin crate");
    let cache = SighashCache::new(&btc_tx);
    let btc_script = ScriptBuf::from(script_code.clone());

    for sighash_type in [
        SIGHASH_ALL as u32,
        SIGHASH_NONE as u32,
        SIGHASH_SINGLE as u32,
        (SIGHASH_ALL | SIGHASH_ANYONECANPAY) as u32,
        (SIGHASH_NONE | SIGHASH_ANYONECANPAY) as u32,
        (SIGHASH_SINGLE | SIGHASH_ANYONECANPAY) as u32,
    ] {
        let ours = tx
            .legacy_sighash(1, &script_code, sighash_type as u8)
            .expect("our sighash");
        let reference = cache
            .legacy_signature_hash(1, btc_script.as_script(), sighash_type)
            .expect("bitcoin crate sighash");
        assert_eq!(
            ours,
            *reference.as_byte_array(),
            "sighash type {sighash_type:#x}"
        );
    }
}

#[test]
fn serialization_roundtrip_with_bitcoin_crate() {
    use bitcoin::consensus::{deserialize, serialize};

    let mut tx = Transaction::new(1, 500_000);
    tx.add_input(TxIn {
        txid: [0xAA; 32],
        vout: 0,
        script_sig: vec![0x00, 0x47, 0x30],
        sequence: 0xffff_ffff,
    });
    tx.add_output(TxOut {
        value: 100_000,
        script_pubkey: vec![0x76, 0xa9, 0x14],
    });

    let our_bytes = tx.serialize();
    let btc_tx: bitcoin::Transaction =
        deserialize(&our_bytes).expect("bitcoin crate should parse our bytes");
    let btc_bytes = serialize(&btc_tx);
    assert_eq!(our_bytes, btc_bytes);
}

#[test]
fn sighash_input_out_of_bounds_is_error() {
    let tx = Transaction::new(1, 0);
    let result = tx.legacy_sighash(0, &[0x51], SIGHASH_ALL);
    assert!(result.is_err());
}
