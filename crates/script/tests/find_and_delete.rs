use script::{find_and_delete, push_data, push_number, count_non_push_opcodes};

#[test]
fn removes_all_occurrences() {
    let sig = [0xAA, 0xBB];
    let mut s = Vec::new();
    s.extend_from_slice(&push_data(&sig));
    s.extend_from_slice(&push_data(&[0xCC, 0xDD]));
    s.extend_from_slice(&push_data(&sig));

    let result = find_and_delete(&s, &sig);
    assert_eq!(result, push_data(&[0xCC, 0xDD]));
}

#[test]
fn removes_pattern_at_start() {
    let sig = [0x01, 0x02];
    let other = [0x03, 0x04];
    let mut s = Vec::new();
    s.extend_from_slice(&push_data(&sig));
    s.extend_from_slice(&push_data(&other));

    let result = find_and_delete(&s, &sig);
    assert_eq!(result, push_data(&other));
}

#[test]
fn removes_pattern_at_end() {
    let sig = [0x01, 0x02];
    let other = [0x03, 0x04];
    let mut s = Vec::new();
    s.extend_from_slice(&push_data(&other));
    s.extend_from_slice(&push_data(&sig));

    let result = find_and_delete(&s, &sig);
    assert_eq!(result, push_data(&other));
}

#[test]
fn no_match_returns_unchanged() {
    let sig = [0x01, 0x02];
    let other = [0x03, 0x04];
    let s = push_data(&other);

    let result = find_and_delete(&s, &sig);
    assert_eq!(result, s);
}

#[test]
fn empty_script_returns_empty() {
    let result = find_and_delete(&[], &[0x01]);
    assert!(result.is_empty());
}

#[test]
fn push_data_small() {
    // Data <= 75 bytes: single-byte length prefix
    let data = [0x42; 5];
    let pushed = push_data(&data);
    assert_eq!(pushed[0], 5); // length
    assert_eq!(&pushed[1..], &data);
}

#[test]
fn push_data_empty() {
    // Empty data → OP_0
    let pushed = push_data(&[]);
    assert_eq!(pushed, vec![0x00]);
}

#[test]
fn push_data_76_bytes() {
    // 76 bytes: OP_PUSHDATA1 prefix
    let data = [0x42; 76];
    let pushed = push_data(&data);
    assert_eq!(pushed[0], 0x4C); // OP_PUSHDATA1
    assert_eq!(pushed[1], 76);
    assert_eq!(&pushed[2..], &data);
}

#[test]
fn push_number_0() {
    assert_eq!(push_number(0), vec![0x00]); // OP_0
}

#[test]
fn push_number_1_to_16() {
    for n in 1..=16i64 {
        let pushed = push_number(n);
        assert_eq!(pushed, vec![0x50 + n as u8]); // OP_1 = 0x51, ..., OP_16 = 0x60
    }
}

#[test]
fn push_number_17() {
    let pushed = push_number(17);
    // 17 = 0x11, fits in one byte, MSB not set
    assert_eq!(pushed, vec![0x01, 0x11]); // push 1 byte: 0x11
}

#[test]
fn push_number_large() {
    let pushed = push_number(300);
    // 300 = 0x012C little-endian = [0x2C, 0x01]
    assert_eq!(pushed, vec![0x02, 0x2C, 0x01]);
}

#[test]
fn push_number_128() {
    // 128 = 0x80 — MSB set, needs 0x00 padding in Script number encoding
    let pushed = push_number(128);
    assert_eq!(pushed, vec![0x02, 0x80, 0x00]);
}

#[test]
fn count_non_push_opcodes_simple() {
    // OP_DUP (0x76) + OP_HASH160 (0xa9) + 20-byte push + OP_EQUALVERIFY (0x88) + OP_CHECKSIG (0xac)
    let mut s = Vec::new();
    s.push(0x76); // OP_DUP
    s.push(0xa9); // OP_HASH160
    s.push(0x14); // push 20 bytes
    s.extend_from_slice(&[0u8; 20]);
    s.push(0x88); // OP_EQUALVERIFY
    s.push(0xac); // OP_CHECKSIG
    assert_eq!(count_non_push_opcodes(&s).unwrap(), 4);
}

#[test]
fn count_non_push_opcodes_only_pushes() {
    // Only push data — zero non-push opcodes
    let mut s = Vec::new();
    s.extend_from_slice(&push_data(&[0x01, 0x02, 0x03]));
    s.extend_from_slice(&push_data(&[0x04, 0x05]));
    assert_eq!(count_non_push_opcodes(&s).unwrap(), 0);
}

#[test]
fn count_non_push_opcodes_op_numbers_not_counted() {
    // OP_0 through OP_16 are NOT counted as non-push opcodes
    let s = vec![0x00, 0x51, 0x52, 0x60]; // OP_0, OP_1, OP_2, OP_16
    assert_eq!(count_non_push_opcodes(&s).unwrap(), 0);
}
