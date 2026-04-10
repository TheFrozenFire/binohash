#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug)]
struct SubsetInput {
    n: u8,
    k: u8,
    index: u32,
}

fuzz_target!(|input: SubsetInput| {
    let n = input.n as usize;
    let k = input.k as usize;
    let index = input.index as u128;

    // Property 1: binomial_coefficient never panics
    let total = subset::binomial_coefficient(n, k);

    // Property 2: nth_combination never panics, returns None for out-of-range
    let combo = subset::nth_combination(n, k, index);
    if k > n || index >= total {
        assert!(combo.is_none());
        return;
    }

    // Property 3: if in range, should produce a valid combination
    let combo = combo.expect("in-range index should produce a combination");
    assert_eq!(combo.len(), k);

    // Property 4: all elements should be in [0, n) and strictly increasing
    for window in combo.windows(2) {
        assert!(window[0] < window[1]);
    }
    if let Some(&last) = combo.last() {
        assert!(last < n);
    }

    // Property 5: combination_index should be the inverse of nth_combination
    let back_index = subset::combination_index(&combo, n);
    assert_eq!(back_index, index, "combination_index should invert nth_combination");

    // Property 6: next/first combination consistency
    if index == 0 {
        let first = subset::first_combination(n, k);
        assert_eq!(first.as_ref(), Some(&combo));
    }
});
