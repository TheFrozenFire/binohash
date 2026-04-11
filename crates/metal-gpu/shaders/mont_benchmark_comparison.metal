// ============================================================
// Direct comparison benchmark: msl-secp256k1's mont_mul_optimised
// verbatim, with secp256k1 constants for 13-bit limbs.
// ============================================================

#define REF_N 20
#define REF_W 13
#define REF_MASK 8191u
#define REF_N0 5425u

struct RefBigInt {
    uint limbs[REF_N];
};

RefBigInt ref_zero() { RefBigInt r; for (int i=0;i<REF_N;i++) r.limbs[i]=0; return r; }

bool ref_gte(RefBigInt a, RefBigInt b) {
    for (int i=REF_N-1;i>=0;i--) {
        if (a.limbs[i]>b.limbs[i]) return true;
        if (a.limbs[i]<b.limbs[i]) return false;
    }
    return true;
}

RefBigInt ref_sub(RefBigInt a, RefBigInt b) {
    RefBigInt r;
    uint borrow=0;
    for (int i=0;i<REF_N;i++) {
        uint diff = a.limbs[i]-b.limbs[i]-borrow;
        if (a.limbs[i]<b.limbs[i]+borrow) {
            diff += (1u<<REF_W);
            borrow=1;
        } else borrow=0;
        r.limbs[i] = diff & REF_MASK;
    }
    return r;
}

RefBigInt ref_conditional_reduce(RefBigInt x, RefBigInt y) {
    if (ref_gte(x,y)) return ref_sub(x,y);
    return x;
}

// Verbatim from msl-secp256k1's mont_mul_optimised
RefBigInt ref_mont_mul_optimised(RefBigInt x, RefBigInt y, RefBigInt p) {
    RefBigInt s = ref_zero();
    for (uint i=0;i<REF_N;i++) {
        uint t = s.limbs[0] + x.limbs[i]*y.limbs[0];
        uint tprime = t & REF_MASK;
        uint qi = (REF_N0*tprime) & REF_MASK;
        uint c = (t + qi*p.limbs[0]) >> REF_W;
        s.limbs[0] = s.limbs[1] + x.limbs[i]*y.limbs[1] + qi*p.limbs[1] + c;
        for (uint j=2;j<REF_N;j++) {
            s.limbs[j-1] = s.limbs[j] + x.limbs[i]*y.limbs[j] + qi*p.limbs[j];
        }
        s.limbs[REF_N-2] = x.limbs[i]*y.limbs[REF_N-1] + qi*p.limbs[REF_N-1];
    }
    uint c=0;
    for (uint i=0;i<REF_N;i++) {
        uint v = s.limbs[i]+c;
        c = v>>REF_W;
        s.limbs[i] = v & REF_MASK;
    }
    return ref_conditional_reduce(s,p);
}

// Benchmark: run N iterations of the reference mont_mul_optimised
kernel void bench_ref_mont_mul(
    const device uchar* seed_be [[buffer(0)]],
    device uchar* out_be        [[buffer(1)]],
    constant uint& iterations   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // Load seed and convert to 13-bit limbs
    uint256 seed = uint256_from_be_device(seed_be);
    seed.d[0] ^= gid;

    // Build p in 13-bit limbs
    RefBigInt p;
    const uint p_data[REF_N] = {
        7215, 8191, 8127, 8191, 8191, 8191, 8191, 8191, 8191, 8191,
        8191, 8191, 8191, 8191, 8191, 8191, 8191, 8191, 8191, 511
    };
    for (int i=0;i<REF_N;i++) p.limbs[i] = p_data[i];

    // Convert seed to 13-bit limbs (raw, not Montgomery form — benchmark
    // only measures multiplication throughput, not correctness)
    RefBigInt a = ref_zero();
    RefBigInt b = ref_zero();
    uint bit_pos = 0;
    for (int i=0;i<REF_N && bit_pos<256;i++) {
        uint word_idx = bit_pos/32;
        uint bit_off = bit_pos%32;
        uint val = seed.d[word_idx] >> bit_off;
        if (bit_off+REF_W>32 && word_idx+1<8) val |= seed.d[word_idx+1]<<(32-bit_off);
        a.limbs[i] = val & REF_MASK;
        b.limbs[i] = (val+1) & REF_MASK;
        bit_pos += REF_W;
    }

    for (uint i=0;i<iterations;i++) {
        a = ref_mont_mul_optimised(a, b, p);
    }

    // Write result (just to prevent dead code elimination)
    if (gid == 0) {
        // Convert back to uint256 for output
        uint256 r = uint256_zero();
        bit_pos = 0;
        for (int i=0;i<REF_N && bit_pos<256;i++) {
            uint word_idx = bit_pos/32;
            uint bit_off = bit_pos%32;
            r.d[word_idx] |= (a.limbs[i]<<bit_off);
            if (bit_off+REF_W>32 && word_idx+1<8) r.d[word_idx+1] |= a.limbs[i]>>(32-bit_off);
            bit_pos += REF_W;
        }
        uint256_to_be_device(r, out_be);
    }
}
