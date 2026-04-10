#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Property 1: is_valid_der_sig should never panic
    let valid = der::is_valid_der_sig(data);

    // Property 2: parse_der_sig should agree with is_valid_der_sig
    let parsed = der::parse_der_sig(data);
    assert_eq!(valid, parsed.is_some(), "is_valid and parse disagree on input");

    // Property 3: if parsed, re-encoding should produce valid DER
    if let Some(p) = parsed {
        let re_encoded = der::encode_der_sig(&p.r, &p.s, p.sighash_type);
        assert!(
            der::is_valid_der_sig(&re_encoded),
            "re-encoded sig should be valid DER"
        );

        // Property 4: re-parsing should roundtrip
        let re_parsed = der::parse_der_sig(&re_encoded).expect("re-encoded should parse");
        assert_eq!(p.r, re_parsed.r, "r should roundtrip");
        assert_eq!(p.s, re_parsed.s, "s should roundtrip");
        assert_eq!(p.sighash_type, re_parsed.sighash_type, "sighash should roundtrip");
    }

    // Property 5: easy_der_predicate should never panic
    der::easy_der_predicate(data);
});
