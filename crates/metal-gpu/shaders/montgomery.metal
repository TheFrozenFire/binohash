// ============================================================
// Montgomery multiplication for secp256k1 field (mod P)
// CIOS "optimised" method with 20 × 13-bit limbs in uint32 words.
//
// 13-bit limbs keep the maximum accumulator value at ~67M, well within
// uint32's 4.3B limit. This avoids the overflow that 16-bit limbs hit
// (where max accumulator = 4.29B + 65K > 2^32).
//
// Adapted from msl-secp256k1 (Geometry Research).
//
// Benchmarked at 18.5ms for 26.2M muls (262K threads × 100 iters)
// vs schoolbook 20.1ms — a 9% improvement. Not yet integrated into
// the main pipeline pending further optimization.
// ============================================================

#define MONT_N 20
#define MONT_W 13
#define MONT_MASK 8191u
#define MONT_N0 5425u

struct mont13 {
    uint d[MONT_N];
};

constant uint MONT_P[MONT_N] = {
    7215, 8191, 8127, 8191, 8191, 8191, 8191, 8191, 8191, 8191,
    8191, 8191, 8191, 8191, 8191, 8191, 8191, 8191, 8191, 511
};

constant uint MONT_R2[MONT_N] = {
    256, 5253, 3, 3908, 0, 128, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

mont13 mont13_zero() { mont13 r; for (int i=0;i<MONT_N;i++) r.d[i]=0; return r; }

bool mont13_gte(mont13 a, mont13 b) {
    for (int i = MONT_N-1; i >= 0; i--) {
        if (a.d[i] > b.d[i]) return true;
        if (a.d[i] < b.d[i]) return false;
    }
    return true;
}

mont13 mont13_sub(mont13 a, mont13 b) {
    mont13 r;
    uint borrow = 0;
    for (int i = 0; i < MONT_N; i++) {
        uint diff = a.d[i] - b.d[i] - borrow;
        if (a.d[i] < b.d[i] + borrow) {
            diff += (1u << MONT_W);
            borrow = 1;
        } else {
            borrow = 0;
        }
        r.d[i] = diff & MONT_MASK;
    }
    return r;
}

mont13 mont13_conditional_reduce(mont13 x) {
    mont13 p; for (int i=0;i<MONT_N;i++) p.d[i] = MONT_P[i];
    if (mont13_gte(x, p)) return mont13_sub(x, p);
    return x;
}

mont13 mont13_mul(mont13 x, mont13 y) {
    mont13 s = mont13_zero();

    for (uint i = 0; i < MONT_N; i++) {
        uint t = s.d[0] + x.d[i] * y.d[0];
        uint tprime = t & MONT_MASK;
        uint qi = (MONT_N0 * tprime) & MONT_MASK;
        uint c = (t + qi * MONT_P[0]) >> MONT_W;
        s.d[0] = s.d[1] + x.d[i] * y.d[1] + qi * MONT_P[1] + c;

        for (uint j = 2; j < MONT_N; j++) {
            s.d[j-1] = s.d[j] + x.d[i] * y.d[j] + qi * MONT_P[j];
        }
        s.d[MONT_N-2] = x.d[i] * y.d[MONT_N-1] + qi * MONT_P[MONT_N-1];
    }

    uint c = 0;
    for (uint i = 0; i < MONT_N; i++) {
        uint v = s.d[i] + c;
        c = v >> MONT_W;
        s.d[i] = v & MONT_MASK;
    }

    return mont13_conditional_reduce(s);
}

mont13 mont13_sqr(mont13 x) { return mont13_mul(x, x); }

mont13 uint256_to_mont13(uint256 a) {
    mont13 r = mont13_zero();
    uint bit_pos = 0;
    for (int i = 0; i < MONT_N && bit_pos < 256; i++) {
        uint word_idx = bit_pos / 32;
        uint bit_off = bit_pos % 32;
        uint val = (a.d[word_idx] >> bit_off);
        if (bit_off + MONT_W > 32 && word_idx + 1 < 8) {
            val |= (a.d[word_idx + 1] << (32 - bit_off));
        }
        r.d[i] = val & MONT_MASK;
        bit_pos += MONT_W;
    }
    return r;
}

uint256 mont13_to_uint256(mont13 a) {
    uint256 r = uint256_zero();
    uint bit_pos = 0;
    for (int i = 0; i < MONT_N && bit_pos < 256; i++) {
        uint word_idx = bit_pos / 32;
        uint bit_off = bit_pos % 32;
        r.d[word_idx] |= (a.d[i] << bit_off);
        if (bit_off + MONT_W > 32 && word_idx + 1 < 8) {
            r.d[word_idx + 1] |= (a.d[i] >> (32 - bit_off));
        }
        bit_pos += MONT_W;
    }
    return r;
}

mont13 mont13_r2() {
    mont13 r; for (int i=0;i<MONT_N;i++) r.d[i] = MONT_R2[i]; return r;
}

mont13 to_mont13(uint256 a) {
    return mont13_mul(uint256_to_mont13(a), mont13_r2());
}

uint256 from_mont13(mont13 a) {
    mont13 one = mont13_zero(); one.d[0] = 1;
    return mont13_to_uint256(mont13_mul(a, one));
}

// ============================================================
// Benchmark and test kernels
// ============================================================

kernel void bench_monty_mul(
    const device uchar* seed_be [[buffer(0)]],
    device uchar* out_be        [[buffer(1)]],
    constant uint& iterations   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint256 seed = uint256_from_be_device(seed_be);
    seed.d[0] ^= gid;
    mont13 a = to_mont13(seed);
    uint256 seed2 = seed; seed2.d[0] += 1;
    mont13 b = to_mont13(seed2);
    for (uint i = 0; i < iterations; i++) a = mont13_mul(a, b);
    if (gid == 0) { uint256 r = from_mont13(a); uint256_to_be_device(r, out_be); }
}

kernel void bench_monty_sqr(
    const device uchar* seed_be [[buffer(0)]],
    device uchar* out_be        [[buffer(1)]],
    constant uint& iterations   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint256 seed = uint256_from_be_device(seed_be);
    seed.d[0] ^= gid;
    mont13 a = to_mont13(seed);
    for (uint i = 0; i < iterations; i++) a = mont13_sqr(a);
    if (gid == 0) { uint256 r = from_mont13(a); uint256_to_be_device(r, out_be); }
}

kernel void test_monty_mul(
    const device uchar* a_be  [[buffer(0)]],
    const device uchar* b_be  [[buffer(1)]],
    device uchar* result_be   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    uint256 a = uint256_from_be_device(a_be);
    uint256 b = uint256_from_be_device(b_be);
    mont13 am = to_mont13(a);
    mont13 bm = to_mont13(b);
    mont13 rm = mont13_mul(am, bm);
    uint256 result = from_mont13(rm);
    uint256_to_be_device(result, result_be);
}
