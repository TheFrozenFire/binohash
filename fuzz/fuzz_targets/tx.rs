#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug)]
struct TxInput {
    version: u32,
    locktime: u32,
    input_txid: [u8; 32],
    input_vout: u32,
    input_sequence: u32,
    output_value: u64,
    output_script: Vec<u8>,
    script_code: Vec<u8>,
    sighash_type: u8,
}

fuzz_target!(|input: TxInput| {
    // Build a transaction
    let mut t = tx::Transaction::new(input.version, input.locktime);
    t.add_input(tx::TxIn {
        txid: input.input_txid,
        vout: input.input_vout,
        script_sig: Vec::new(),
        sequence: input.input_sequence,
    });
    t.add_output(tx::TxOut {
        value: input.output_value,
        script_pubkey: input.output_script,
    });

    // Property 1: serialize should never panic
    let serialized = t.serialize();
    assert!(!serialized.is_empty());

    // Property 2: legacy_sighash should never panic (may return Err)
    let _ = t.legacy_sighash(0, &input.script_code, input.sighash_type);

    // Property 3: sighash should be deterministic
    if let Ok(hash1) = t.legacy_sighash(0, &input.script_code, input.sighash_type) {
        let hash2 = t.legacy_sighash(0, &input.script_code, input.sighash_type).unwrap();
        assert_eq!(hash1, hash2);
    }

    // Property 4: out-of-bounds input_index returns Err
    assert!(t.legacy_sighash(1, &input.script_code, input.sighash_type).is_err());
});
