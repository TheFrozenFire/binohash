/// Iterator over all k-element subsets of {0, ..., n-1} in lexicographic order.
#[derive(Debug, Clone)]
pub struct CombinationIter {
    n: usize,
    k: usize,
    current: Option<Vec<usize>>,
}

impl CombinationIter {
    pub fn new(n: usize, k: usize) -> Self {
        let current = first_combination(n, k);
        Self { n, k, current }
    }
}

impl Iterator for CombinationIter {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current.clone()?;
        if self.k == 0 {
            // The single empty-set combination — yield it once, then done
            self.current = None;
            return Some(current);
        }

        self.current = next_combination(&current, self.n);
        Some(current)
    }
}

/// Return the first combination of k elements from {0, ..., n-1}.
///
/// Returns `Some(vec![0, 1, ..., k-1])` for k <= n, or `Some(vec![])` for k == 0.
/// Returns `None` when k > n (no valid combinations).
pub fn first_combination(n: usize, k: usize) -> Option<Vec<usize>> {
    if k == 0 {
        Some(Vec::new())
    } else if k > n {
        None
    } else {
        Some((0..k).collect())
    }
}

/// Advance to the next combination in lexicographic order.
///
/// Given the current combination (a sorted slice of indices into {0, ..., n-1}),
/// returns the next combination, or `None` if this was the last one.
pub fn next_combination(current: &[usize], n: usize) -> Option<Vec<usize>> {
    let k = current.len();
    if k == 0 {
        return None;
    }
    let mut next = current.to_vec();
    // Find the rightmost element that can be incremented
    let mut i = k;
    while i > 0 {
        i -= 1;
        if next[i] != i + n - k {
            next[i] += 1;
            // Reset all elements to the right
            for j in i + 1..k {
                next[j] = next[j - 1] + 1;
            }
            return Some(next);
        }
    }
    None
}

/// Compute the nth combination in lexicographic order using the combinatorial
/// number system.
///
/// Given an index in [0, C(n,k)), returns the corresponding k-element subset.
/// Returns `None` if the index is out of range or k > n.
///
/// This enables O(k) random access into the combination space — critical for
/// parallel partitioning of the C(150,9) search space across workers.
pub fn nth_combination(n: usize, k: usize, mut index: u128) -> Option<Vec<usize>> {
    if k == 0 {
        return if index == 0 { Some(Vec::new()) } else { None };
    }
    if k > n || index >= binomial_coefficient(n, k) {
        return None;
    }

    let mut result = Vec::with_capacity(k);
    let mut next = 0usize;
    let mut remaining_k = k;

    for _ in 0..k {
        let mut c = next;
        loop {
            let count = binomial_coefficient(n - 1 - c, remaining_k - 1);
            if index < count {
                result.push(c);
                next = c + 1;
                remaining_k -= 1;
                break;
            }
            index -= count;
            c += 1;
        }
    }

    Some(result)
}

/// Compute the lexicographic index of a combination (inverse of `nth_combination`).
///
/// Returns the index in [0, C(n,k)) for the given combination.
pub fn combination_index(combo: &[usize], n: usize) -> u128 {
    let k = combo.len();
    let mut index: u128 = 0;
    let mut start = 0usize;

    for (i, &c) in combo.iter().enumerate() {
        let remaining_k = k - i;
        for j in start..c {
            index += binomial_coefficient(n - 1 - j, remaining_k - 1);
        }
        start = c + 1;
    }

    index
}

/// Compute the binomial coefficient C(n, k) = n! / (k! * (n-k)!).
///
/// Returns 0 when k > n. Uses checked u128 arithmetic, returning `u128::MAX`
/// on overflow (which is safe — callers compare against this for out-of-range checks).
pub fn binomial_coefficient(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    // Use the smaller of k and n-k to minimize iterations
    let k = k.min(n - k);
    let mut result: u128 = 1;
    for i in 0..k {
        // Multiply first, then divide. The product C(n,i) * (n-i) / (i+1) is
        // always exact because C(n, i+1) = C(n, i) * (n-i) / (i+1) is an integer.
        result = match result.checked_mul((n - i) as u128) {
            Some(v) => v / (i + 1) as u128,
            None => return u128::MAX, // overflow — value is astronomically large
        };
    }
    result
}
