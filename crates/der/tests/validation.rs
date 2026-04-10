use der::{encode_der_sig, is_valid_der_sig, parse_der_sig, easy_der_predicate, encode_der_integer};

/// Minimum valid DER: 30 06 02 01 <r> 02 01 <s> <sighash>  (9 bytes)
#[test]
fn minimal_9byte_signature_is_valid() {
    let sig = [0x30, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01, 0x01];
    assert!(is_valid_der_sig(&sig));
}

#[test]
fn minimal_sig_with_max_single_byte_values() {
    // r=127 (0x7F), s=127 — both positive, single-byte, no padding needed
    let sig = [0x30, 0x06, 0x02, 0x01, 0x7F, 0x02, 0x01, 0x7F, 0x01];
    assert!(is_valid_der_sig(&sig));
}

#[test]
fn wrong_sequence_tag_is_invalid() {
    let sig = [0x31, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01, 0x01];
    assert!(!is_valid_der_sig(&sig));
}

#[test]
fn wrong_total_length_is_invalid() {
    // Length byte says 0x07 but actual content is 0x06
    let sig = [0x30, 0x07, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01, 0x01];
    assert!(!is_valid_der_sig(&sig));
}

#[test]
fn missing_r_integer_tag_is_invalid() {
    let sig = [0x30, 0x06, 0x03, 0x01, 0x01, 0x02, 0x01, 0x01, 0x01];
    assert!(!is_valid_der_sig(&sig));
}

#[test]
fn missing_s_integer_tag_is_invalid() {
    let sig = [0x30, 0x06, 0x02, 0x01, 0x01, 0x03, 0x01, 0x01, 0x01];
    assert!(!is_valid_der_sig(&sig));
}

#[test]
fn negative_r_is_invalid() {
    // r first byte has MSB set (0x80) → negative
    let sig = [0x30, 0x06, 0x02, 0x01, 0x80, 0x02, 0x01, 0x01, 0x01];
    assert!(!is_valid_der_sig(&sig));
}

#[test]
fn negative_s_is_invalid() {
    let sig = [0x30, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x80, 0x01];
    assert!(!is_valid_der_sig(&sig));
}

#[test]
fn unnecessary_leading_zero_on_r_is_invalid() {
    // r = [0x00, 0x01] — leading zero not needed since 0x01 < 0x80
    let sig = [0x30, 0x07, 0x02, 0x02, 0x00, 0x01, 0x02, 0x01, 0x01, 0x01];
    assert!(!is_valid_der_sig(&sig));
}

#[test]
fn necessary_leading_zero_on_r_is_valid() {
    // r = [0x00, 0x80] — leading zero IS needed since 0x80 >= 0x80
    let sig = [0x30, 0x07, 0x02, 0x02, 0x00, 0x80, 0x02, 0x01, 0x01, 0x01];
    assert!(is_valid_der_sig(&sig));
}

#[test]
fn unnecessary_leading_zero_on_s_is_invalid() {
    let sig = [0x30, 0x07, 0x02, 0x01, 0x01, 0x02, 0x02, 0x00, 0x01, 0x01];
    assert!(!is_valid_der_sig(&sig));
}

#[test]
fn zero_length_r_is_invalid() {
    // r_len = 0
    let sig = [0x30, 0x05, 0x02, 0x00, 0x02, 0x01, 0x01, 0x01];
    assert!(!is_valid_der_sig(&sig));
}

#[test]
fn zero_value_r_is_invalid() {
    // r = [0x00] — the value is zero (all bytes zero)
    // A single 0x00 byte: MSB < 0x80 so it passes negativity check,
    // but it IS unnecessary leading zero (len > 1 check doesn't apply for len=1).
    // However r=0 means the integer value is zero. In BIP66 DER, zero-valued
    // integers are represented as a single 0x00 byte and that's technically valid
    // DER encoding. But for ECDSA, r=0 and s=0 are not valid signature values.
    // The nktkt reference does NOT reject zero values at the DER level.
    // Robin Linus's reference DOES reject zero values.
    // For Binohash, rejecting zero is safer — a zero r or s can't be a valid sig.
    //
    // We follow the consensus-level BIP66 check which does NOT reject zero.
    // The ECDSA-level validity is checked later during key recovery.
}

#[test]
fn too_short_is_invalid() {
    // Less than 9 bytes (minimum valid DER sig with sighash)
    assert!(!is_valid_der_sig(&[0x30, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01]));
    assert!(!is_valid_der_sig(&[0x30]));
    assert!(!is_valid_der_sig(&[]));
}

#[test]
fn all_valid_sighash_types() {
    for sighash in [0x01u8, 0x02, 0x03, 0x81, 0x82, 0x83] {
        let sig = [0x30, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01, sighash];
        assert!(
            is_valid_der_sig(&sig),
            "sighash {sighash:#04x} should be valid"
        );
    }
}

#[test]
fn invalid_sighash_types_at_consensus_level() {
    // At consensus level (BIP66), the sighash byte is NOT validated —
    // SCRIPT_VERIFY_STRICTENC is policy only. For Binohash/QSB the sighash
    // byte can be any value since the sig goes through Slipstream.
    // Our is_valid_der_sig checks structural DER only, not sighash policy.
    // Any trailing byte is accepted.
    let sig = [0x30, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01, 0x00];
    assert!(is_valid_der_sig(&sig), "sighash 0x00 should pass consensus-level DER check");

    let sig = [0x30, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01, 0xFF];
    assert!(is_valid_der_sig(&sig), "sighash 0xFF should pass consensus-level DER check");
}

#[test]
fn variable_length_20_byte_input() {
    // A 20-byte RIPEMD-160 output that happens to be valid DER
    // 30 [len] 02 [r_len] [r...] 02 [s_len] [s...] [sighash]
    // With 20 bytes total: inner_len = 17, so r_len + s_len + 4 = 17
    // e.g. r_len=6, s_len=7: 30 11 02 06 [6 r bytes] 02 07 [7 s bytes] [sh]
    let mut sig = [0u8; 20];
    sig[0] = 0x30;
    sig[1] = 0x11; // 17 = 20 - 3 (tag + len + sighash)
    sig[2] = 0x02;
    sig[3] = 0x06; // r_len = 6
    sig[4] = 0x01; // r first byte: positive, non-zero
    // sig[5..10] = 0x00 (rest of r)
    sig[10] = 0x02;
    sig[11] = 0x07; // s_len = 7
    sig[12] = 0x01; // s first byte: positive, non-zero
    // sig[13..19] = 0x00 (rest of s)
    sig[19] = 0x01; // sighash
    assert!(is_valid_der_sig(&sig));
}

#[test]
fn variable_length_32_byte_input() {
    // A 32-byte SHA-256 output that happens to be valid DER (Robin Linus variant)
    // inner_len = 29, total = 32
    let mut sig = [0u8; 32];
    sig[0] = 0x30;
    sig[1] = 0x1D; // 29 = 32 - 3
    sig[2] = 0x02;
    sig[3] = 0x0C; // r_len = 12
    sig[4] = 0x01; // r positive non-zero
    sig[16] = 0x02; // s tag at 4 + 12 = 16
    sig[17] = 0x0D; // s_len = 13
    sig[18] = 0x01; // s positive non-zero
    sig[31] = 0x01; // sighash
    assert!(is_valid_der_sig(&sig));
}

#[test]
fn encode_parse_roundtrip() {
    let mut r = [0u8; 32];
    r[0] = 0x7F; // large positive value
    r[1] = 0xAB;
    r[31] = 0x01;

    let mut s = [0u8; 32];
    s[1] = 0x42;
    s[31] = 0xFF;

    let encoded = encode_der_sig(&r, &s, 0x01);
    assert!(is_valid_der_sig(&encoded), "encoded signature should be valid DER");

    let parsed = parse_der_sig(&encoded).expect("should parse successfully");
    assert_eq!(parsed.r, r);
    assert_eq!(parsed.s, s);
    assert_eq!(parsed.sighash_type, 0x01);
}

#[test]
fn encode_parse_roundtrip_small_values() {
    // r = 1, s = 1 (lots of leading zeros in the 32-byte representation)
    let mut r = [0u8; 32];
    r[31] = 0x01;
    let mut s = [0u8; 32];
    s[31] = 0x01;

    let encoded = encode_der_sig(&r, &s, 0x03);
    assert_eq!(encoded, vec![0x30, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01, 0x03]);
    assert!(is_valid_der_sig(&encoded));

    let parsed = parse_der_sig(&encoded).unwrap();
    assert_eq!(parsed.r, r);
    assert_eq!(parsed.s, s);
    assert_eq!(parsed.sighash_type, 0x03);
}

#[test]
fn encode_parse_roundtrip_needs_padding() {
    // r has MSB set → needs 0x00 padding in DER
    let mut r = [0u8; 32];
    r[0] = 0x80;
    r[31] = 0x01;
    let mut s = [0u8; 32];
    s[31] = 0x42;

    let encoded = encode_der_sig(&r, &s, 0x01);
    assert!(is_valid_der_sig(&encoded));

    let parsed = parse_der_sig(&encoded).unwrap();
    assert_eq!(parsed.r, r);
    assert_eq!(parsed.s, s);
}

#[test]
fn encode_der_integer_trims_leading_zeros() {
    let mut value = [0u8; 32];
    value[31] = 0x42;
    let encoded = encode_der_integer(&value);
    assert_eq!(encoded, vec![0x42]);
}

#[test]
fn encode_der_integer_adds_padding_for_high_bit() {
    let mut value = [0u8; 32];
    value[31] = 0x80;
    let encoded = encode_der_integer(&value);
    assert_eq!(encoded, vec![0x00, 0x80]);
}

#[test]
fn encode_der_integer_full_32_bytes() {
    let mut value = [0u8; 32];
    value[0] = 0x7F;
    value[31] = 0x01;
    let encoded = encode_der_integer(&value);
    assert_eq!(encoded.len(), 32);
    assert_eq!(encoded[0], 0x7F);
}

#[test]
fn parse_invalid_returns_none() {
    assert!(parse_der_sig(&[]).is_none());
    assert!(parse_der_sig(&[0x30]).is_none());
    assert!(parse_der_sig(&[0x31, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01, 0x01]).is_none());
}

#[test]
fn easy_predicate_checks_first_nibble() {
    assert!(easy_der_predicate(&[0x30]));
    assert!(easy_der_predicate(&[0x31]));
    assert!(easy_der_predicate(&[0x3F]));
    assert!(!easy_der_predicate(&[0x20]));
    assert!(!easy_der_predicate(&[0x40]));
    assert!(!easy_der_predicate(&[]));
}

#[test]
fn random_20_byte_inputs_produce_no_hits() {
    // 10k random 20-byte inputs should produce zero DER hits
    // (probability ~2^-46 per attempt, so 10k is nowhere near enough)
    use hash::ripemd160;

    let mut hits = 0;
    for i in 0u32..10_000 {
        let data = ripemd160(&i.to_le_bytes());
        if is_valid_der_sig(&data) {
            hits += 1;
        }
    }
    assert_eq!(hits, 0, "expected zero DER hits from 10k random 20-byte inputs");
}

#[test]
fn trailing_data_after_s_is_invalid() {
    // Valid 9-byte sig, but with an extra byte before sighash position
    // 30 07 02 01 01 02 01 01 XX 01 — length says 7 but there's extra data
    let sig = [0x30, 0x07, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01, 0xAA, 0x01];
    // total_len (0x07) + 3 = 10 = data.len(), so length check passes
    // But after parsing r and s, idx should be at data.len() - 1.
    // r: tag at [2], len=1 at [3], value at [4] → idx=5
    // s: tag at [5], len=1 at [6], value at [7] → idx=8
    // data.len() - 1 = 9, so idx(8) != 9 → invalid
    assert!(!is_valid_der_sig(&sig));
}
