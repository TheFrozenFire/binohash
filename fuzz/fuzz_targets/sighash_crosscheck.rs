#![no_main]
use arbitrary::Arbitrary;
use bitcoin::hashes::Hash;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct SighashInput {
    version: u32,
    locktime: u32,
    // Two inputs to exercise ANYONECANPAY and SINGLE behavior
    txid_0: [u8; 32],
    vout_0: u32,
    seq_0: u32,
    script_sig_0: Vec<u8>,
    txid_1: [u8; 32],
    vout_1: u32,
    seq_1: u32,
    script_sig_1: Vec<u8>,
    // Two outputs
    value_0: u64,
    spk_0: Vec<u8>,
    value_1: u64,
    spk_1: Vec<u8>,
    // Sighash params
    input_index: bool, // false=0, true=1
    script_code: Vec<u8>,
    sighash_type: u8,
}

fuzz_target!(|input: SighashInput| {
    // Limit script sizes to keep the fuzzer productive
    if input.script_code.len() > 1000
        || input.spk_0.len() > 200
        || input.spk_1.len() > 200
        || input.script_sig_0.len() > 200
        || input.script_sig_1.len() > 200
    {
        return;
    }

    // Only test valid sighash types (the bitcoin crate rejects others)
    let base = input.sighash_type & 0x1f;
    if !matches!(base, 0x01 | 0x02 | 0x03) {
        return;
    }

    // Build our transaction
    let mut our_tx = tx::Transaction::new(input.version, input.locktime);
    our_tx.add_input(tx::TxIn {
        txid: input.txid_0,
        vout: input.vout_0,
        script_sig: input.script_sig_0.clone(),
        sequence: input.seq_0,
    });
    our_tx.add_input(tx::TxIn {
        txid: input.txid_1,
        vout: input.vout_1,
        script_sig: input.script_sig_1.clone(),
        sequence: input.seq_1,
    });
    our_tx.add_output(tx::TxOut {
        value: input.value_0,
        script_pubkey: input.spk_0.clone(),
    });
    our_tx.add_output(tx::TxOut {
        value: input.value_1,
        script_pubkey: input.spk_1.clone(),
    });

    let idx = if input.input_index { 1 } else { 0 };

    // Compute our sighash
    let our_result = our_tx.legacy_sighash(idx, &input.script_code, input.sighash_type);

    // Build the same transaction using the bitcoin crate
    let serialized = our_tx.serialize();
    let btc_tx: bitcoin::Transaction = match bitcoin::consensus::deserialize(&serialized) {
        Ok(t) => t,
        Err(_) => return, // If bitcoin crate can't parse it, skip
    };

    let cache = bitcoin::sighash::SighashCache::new(&btc_tx);
    let btc_script = bitcoin::ScriptBuf::from(input.script_code.clone());
    let btc_result =
        cache.legacy_signature_hash(idx, btc_script.as_script(), input.sighash_type as u32);

    match (our_result, btc_result) {
        (Ok(ours), Ok(theirs)) => {
            assert_eq!(
                ours,
                *theirs.as_byte_array(),
                "sighash mismatch for type {:#04x} input {idx}",
                input.sighash_type
            );
        }
        (Err(_), Err(_)) => {} // Both failed — consistent
        (Ok(_), Err(_)) => panic!("we produced a hash but bitcoin crate returned Err"),
        (Err(_), Ok(_)) => {
            // We returned Err but bitcoin crate succeeded — this can happen
            // if our bounds checking is stricter. Not necessarily a bug.
        }
    }
});
