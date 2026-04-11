// ============================================================
// Batch field inversion using Montgomery's trick
//
// N threads in a threadgroup cooperate to compute N inversions
// using only 1 actual field_inv + 3(N-1) field_muls.
//
// vs per-thread: N × field_inv = N × 271 field_muls
// batch:         271 + 3(N-1) field_muls total
// For N=256:     271 + 765 = 1036 total vs 69,376 = 67× cheaper
// ============================================================

#define BATCH_SIZE 256  // must match threadgroup size

// Per-thread inversion baseline: each thread inverts independently
kernel void bench_inv_per_thread(
    const device uchar* seed_be [[buffer(0)]],
    device uchar* out_be        [[buffer(1)]],
    constant uint& iterations   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint256 seed = uint256_from_be_device(seed_be);
    seed.d[0] ^= gid;
    // Ensure non-zero
    if (uint256_is_zero(seed)) seed = uint256_one();

    uint256 val = seed;
    for (uint i = 0; i < iterations; i++) {
        val = field_inv(val);
    }
    if (gid == 0) uint256_to_be_device(val, out_be);
}

// Batch inversion v1: sequential (thread 0 does all work)
kernel void bench_inv_batch(
    const device uchar* seed_be [[buffer(0)]],
    device uchar* out_be        [[buffer(1)]],
    constant uint& iterations   [[buffer(2)]],
    uint gid                    [[thread_position_in_grid]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint tg_size                [[threads_per_threadgroup]]
) {
    threadgroup uint256 z_vals[BATCH_SIZE];
    threadgroup uint256 prefix[BATCH_SIZE];
    threadgroup uint256 inv_out[BATCH_SIZE];

    uint n = min(tg_size, (uint)BATCH_SIZE);

    uint256 seed = uint256_from_be_device(seed_be);
    seed.d[0] ^= gid;
    if (uint256_is_zero(seed)) seed = uint256_one();

    uint256 my_val = seed;

    for (uint iter = 0; iter < iterations; iter++) {
        z_vals[tid] = my_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            prefix[0] = z_vals[0];
            for (uint i = 1; i < n; i++) {
                prefix[i] = field_mul(prefix[i-1], z_vals[i]);
            }
            uint256 inv = field_inv(prefix[n-1]);
            for (uint i = n - 1; i > 0; i--) {
                inv_out[i] = field_mul(inv, prefix[i-1]);
                inv = field_mul(inv, z_vals[i]);
            }
            inv_out[0] = inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        my_val = inv_out[tid];
    }

    if (gid == 0) uint256_to_be_device(my_val, out_be);
}

// Batch inversion v2: parallel prefix tree
// All threads participate in building the prefix product tree (log2(N) steps),
// then one thread inverts, then all threads scatter back.
kernel void bench_inv_batch_tree(
    const device uchar* seed_be [[buffer(0)]],
    device uchar* out_be        [[buffer(1)]],
    constant uint& iterations   [[buffer(2)]],
    uint gid                    [[thread_position_in_grid]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint tg_size                [[threads_per_threadgroup]]
) {
    threadgroup uint256 tree[BATCH_SIZE];
    threadgroup uint256 originals[BATCH_SIZE];

    uint n = min(tg_size, (uint)BATCH_SIZE);

    uint256 seed = uint256_from_be_device(seed_be);
    seed.d[0] ^= gid;
    if (uint256_is_zero(seed)) seed = uint256_one();

    uint256 my_val = seed;

    for (uint iter = 0; iter < iterations; iter++) {
        originals[tid] = my_val;
        tree[tid] = my_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Up-sweep: parallel prefix products (reduce phase)
        for (uint stride = 1; stride < n; stride <<= 1) {
            uint idx = (tid + 1) * (stride << 1) - 1;
            if (idx < n) {
                tree[idx] = field_mul(tree[idx - stride], tree[idx]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Thread 0 inverts the total product
        if (tid == 0) {
            tree[n - 1] = field_inv(tree[n - 1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Down-sweep: distribute inverses
        for (uint stride = n >> 1; stride >= 1; stride >>= 1) {
            uint idx = (tid + 1) * (stride << 1) - 1;
            if (idx < n) {
                uint256 left = tree[idx - stride];
                tree[idx - stride] = tree[idx];
                tree[idx] = field_mul(tree[idx], left);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        my_val = tree[tid];
    }

    if (gid == 0) uint256_to_be_device(my_val, out_be);
}
