#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|value: i64| {
    // Only test non-negative values — Bitcoin Script numbers in Binohash
    // are always non-negative indices/positions.
    if value < 0 {
        return;
    }

    // Property 1: push_number should never panic for non-negative values
    let encoded = script::push_number(value);
    assert!(!encoded.is_empty());

    // Property 2: output should be valid Script (parseable by count_non_push_opcodes)
    let opcount = script::count_non_push_opcodes(&encoded);
    assert!(opcount.is_ok(), "push_number output should be valid script");

    // Property 3: push_number should produce zero non-push opcodes
    // (it's always a push operation)
    assert_eq!(opcount.unwrap(), 0, "push_number should only produce push ops");

    // Property 4: specific known encodings
    match value {
        0 => assert_eq!(encoded, vec![0x00]), // OP_0
        1..=16 => {
            assert_eq!(encoded.len(), 1);
            assert_eq!(encoded[0], 0x50 + value as u8); // OP_1..OP_16
        }
        _ => {
            // For values > 16, should be a push of minimal little-endian encoding
            assert!(encoded.len() >= 2); // at least push-len + 1 byte
            let push_len = encoded[0] as usize;
            assert_eq!(encoded.len(), 1 + push_len);

            // Decode the script number back
            let num_bytes = &encoded[1..];
            let mut result: i64 = 0;
            for (i, &byte) in num_bytes.iter().enumerate() {
                result |= (byte as i64) << (8 * i);
            }
            // If the last byte had a sign bit set, there would be a padding byte
            // For positive numbers, the decoded value should match
            assert_eq!(result, value, "push_number({value}) decodes to {result}");
        }
    }
});
