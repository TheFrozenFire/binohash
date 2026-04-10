use subset::{CombinationIter, binomial_coefficient, first_combination, next_combination};

#[test]
fn c_5_3_produces_10_subsets_in_lex_order() {
    let combos: Vec<Vec<usize>> = CombinationIter::new(5, 3).collect();
    assert_eq!(
        combos,
        vec![
            vec![0, 1, 2],
            vec![0, 1, 3],
            vec![0, 1, 4],
            vec![0, 2, 3],
            vec![0, 2, 4],
            vec![0, 3, 4],
            vec![1, 2, 3],
            vec![1, 2, 4],
            vec![1, 3, 4],
            vec![2, 3, 4],
        ]
    );
}

#[test]
fn next_combination_matches_iterator() {
    let all: Vec<Vec<usize>> = CombinationIter::new(6, 3).collect();
    let mut walked = Vec::new();
    let mut current = first_combination(6, 3);
    while let Some(combo) = current {
        walked.push(combo.clone());
        current = next_combination(&combo, 6);
    }
    assert_eq!(walked, all);
}

#[test]
fn c_n_0_yields_one_empty_set() {
    let combos: Vec<Vec<usize>> = CombinationIter::new(5, 0).collect();
    assert_eq!(combos, vec![Vec::<usize>::new()]);
}

#[test]
fn c_n_n_yields_one_full_set() {
    let combos: Vec<Vec<usize>> = CombinationIter::new(4, 4).collect();
    assert_eq!(combos, vec![vec![0, 1, 2, 3]]);
}

#[test]
fn c_0_0_yields_one_empty_set() {
    let combos: Vec<Vec<usize>> = CombinationIter::new(0, 0).collect();
    assert_eq!(combos, vec![Vec::<usize>::new()]);
}

#[test]
fn c_k_greater_than_n_yields_nothing() {
    let combos: Vec<Vec<usize>> = CombinationIter::new(3, 5).collect();
    assert!(combos.is_empty());
}

#[test]
fn c_1_1_yields_single_element() {
    let combos: Vec<Vec<usize>> = CombinationIter::new(1, 1).collect();
    assert_eq!(combos, vec![vec![0]]);
}

#[test]
fn count_matches_binomial_coefficient() {
    let count = CombinationIter::new(10, 4).count();
    assert_eq!(count, binomial_coefficient(10, 4) as usize);
}

#[test]
fn binomial_coefficient_known_values() {
    assert_eq!(binomial_coefficient(5, 3), 10);
    assert_eq!(binomial_coefficient(10, 0), 1);
    assert_eq!(binomial_coefficient(10, 10), 1);
    assert_eq!(binomial_coefficient(10, 1), 10);
    assert_eq!(binomial_coefficient(20, 10), 184_756);
    assert_eq!(binomial_coefficient(0, 0), 1);
    assert_eq!(binomial_coefficient(3, 5), 0);
}

#[test]
fn binomial_coefficient_large() {
    // C(150, 9) = 82,947,113,349,100 — the QSB Config A total subset count
    assert_eq!(binomial_coefficient(150, 9), 82_947_113_349_100);
}

#[test]
fn first_combination_returns_none_when_k_exceeds_n() {
    assert!(first_combination(3, 5).is_none());
}

#[test]
fn next_combination_returns_none_at_end() {
    // Last combination of C(4, 2) is [2, 3]
    assert!(next_combination(&[2, 3], 4).is_none());
}

#[test]
fn next_combination_on_empty_returns_none() {
    assert!(next_combination(&[], 5).is_none());
}

#[test]
fn all_subsets_are_sorted_internally() {
    for combo in CombinationIter::new(8, 4) {
        for window in combo.windows(2) {
            assert!(window[0] < window[1], "subset {combo:?} is not sorted");
        }
    }
}

#[test]
fn subsets_are_in_strictly_increasing_lex_order() {
    let combos: Vec<Vec<usize>> = CombinationIter::new(7, 3).collect();
    for window in combos.windows(2) {
        assert!(
            window[0] < window[1],
            "{:?} should come before {:?}",
            window[0],
            window[1]
        );
    }
}
