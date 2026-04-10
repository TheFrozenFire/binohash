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

/// Compute the binomial coefficient C(n, k) = n! / (k! * (n-k)!).
///
/// Returns 0 when k > n. Uses u128 to handle large values like C(150, 9).
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
        result = result * (n - i) as u128;
        result /= (i + 1) as u128;
    }
    result
}
