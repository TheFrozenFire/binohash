#![no_main]
use libfuzzer_sys::fuzz_target;

/// secp256k1 curve order N
const N: [u8; 32] = [
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xfe, 0xba, 0xae, 0xdc, 0xe6, 0xaf, 0x48, 0xa0, 0x3b, 0xbf, 0xd2, 0x5e, 0x8c, 0xd0, 0x36,
    0x41, 0x41,
];

fuzz_target!(|data: [u8; 32]| {
    let valid = ecdsa_recovery::is_valid_scalar(&data);

    // Property 1: zero should never be valid
    if data.iter().all(|b| *b == 0) {
        assert!(!valid, "zero should not be a valid scalar");
        return;
    }

    // Property 2: values >= N should not be valid
    // Compare big-endian: find first differing byte
    let cmp = data.iter().zip(N.iter()).find_map(|(a, b)| {
        if a != b {
            Some(a.cmp(b))
        } else {
            None
        }
    });

    match cmp {
        Some(std::cmp::Ordering::Greater) => {
            assert!(!valid, "value > N should not be valid");
        }
        Some(std::cmp::Ordering::Less) => {
            assert!(valid, "non-zero value < N should be valid");
        }
        None => {
            // data == N
            assert!(!valid, "N itself should not be a valid scalar");
        }
        Some(std::cmp::Ordering::Equal) => unreachable!(),
    }

    // Property 3: if valid, it should work as a secp256k1 secret key
    if valid {
        let result = secp256k1::SecretKey::from_byte_array(data);
        assert!(
            result.is_ok(),
            "is_valid_scalar returned true but SecretKey::from_byte_array failed"
        );
    }
});
