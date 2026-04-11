// Benchmark: N iterations of HASH160 (SHA-256 + RIPEMD-160 of 33-byte compressed pubkey)
kernel void bench_hash160(
    const device uchar* seed_be [[buffer(0)]],
    device uchar* out_be        [[buffer(1)]],
    constant uint& iterations   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint256 seed = uint256_from_be_device(seed_be);
    seed.d[0] ^= gid;

    uchar compressed[33];
    compressed[0] = 0x02;
    uint256_to_be_thread(seed, compressed + 1);

    uchar h160[20];
    for (uint i = 0; i < iterations; i++) {
        hash160_33bytes(compressed, h160);
        // Feed result back to prevent dead-code elimination
        compressed[1] ^= h160[0];
    }

    if (gid == 0) {
        for (int i = 0; i < 20; i++) out_be[i] = h160[i];
    }
}

// Benchmark: just the RIPEMD-160 portion (32-byte input)
kernel void bench_ripemd160(
    const device uchar* seed_be [[buffer(0)]],
    device uchar* out_be        [[buffer(1)]],
    constant uint& iterations   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint256 seed = uint256_from_be_device(seed_be);
    seed.d[0] ^= gid;

    uchar input[32];
    uint256_to_be_thread(seed, input);

    uchar output[20];
    for (uint i = 0; i < iterations; i++) {
        ripemd160_32bytes(input, output);
        input[0] ^= output[0];
    }

    if (gid == 0) {
        for (int i = 0; i < 20; i++) out_be[i] = output[i];
    }
}
