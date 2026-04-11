// ============================================================
// EC operation benchmark: their exact code vs ours
//
// This file contains the msl-secp256k1 stack VERBATIM (with
// secp256k1 13-bit constants inlined) and benchmark kernels
// that exercise the same EC workload.
//
// Workload: 16 Jacobian point additions (simulating GTable accumulation)
// ============================================================

// --- msl-secp256k1 constants (secp256k1, 13-bit limbs) ---
#define GEO_NUM_LIMBS 20
#define GEO_NUM_LIMBS_WIDE 21
#define GEO_LOG_LIMB_SIZE 13
#define GEO_TWO_POW_WORD_SIZE 8192
#define GEO_MASK 8191
#define GEO_N0 5425

// --- msl-secp256k1 bigint (verbatim, prefixed to avoid collision) ---

struct GeoBigInt {
    uint limbs[GEO_NUM_LIMBS];
};

struct GeoBigIntWide {
    uint limbs[GEO_NUM_LIMBS_WIDE];
};

GeoBigInt geo_bigint_zero() {
    GeoBigInt s;
    for (uint i = 0; i < GEO_NUM_LIMBS; i++) s.limbs[i] = 0;
    return s;
}

GeoBigInt geo_bigint_add_unsafe(GeoBigInt lhs, GeoBigInt rhs) {
    GeoBigInt result;
    uint carry = 0;
    for (uint i = 0; i < GEO_NUM_LIMBS; i++) {
        uint c = lhs.limbs[i] + rhs.limbs[i] + carry;
        result.limbs[i] = c & GEO_MASK;
        carry = c >> GEO_LOG_LIMB_SIZE;
    }
    return result;
}

GeoBigIntWide geo_bigint_add_wide(GeoBigInt lhs, GeoBigInt rhs) {
    GeoBigIntWide result;
    uint carry = 0;
    for (uint i = 0; i < GEO_NUM_LIMBS; i++) {
        uint c = lhs.limbs[i] + rhs.limbs[i] + carry;
        result.limbs[i] = c & GEO_MASK;
        carry = c >> GEO_LOG_LIMB_SIZE;
    }
    result.limbs[GEO_NUM_LIMBS] = carry;
    return result;
}

GeoBigInt geo_bigint_sub(GeoBigInt lhs, GeoBigInt rhs) {
    uint borrow = 0;
    GeoBigInt res;
    for (uint i = 0; i < GEO_NUM_LIMBS; i++) {
        res.limbs[i] = lhs.limbs[i] - rhs.limbs[i] - borrow;
        if (lhs.limbs[i] < (rhs.limbs[i] + borrow)) {
            res.limbs[i] = res.limbs[i] + GEO_TWO_POW_WORD_SIZE;
            borrow = 1;
        } else {
            borrow = 0;
        }
    }
    return res;
}

GeoBigIntWide geo_bigint_sub_wide(GeoBigIntWide lhs, GeoBigIntWide rhs) {
    uint borrow = 0;
    GeoBigIntWide res;
    for (uint i = 0; i < GEO_NUM_LIMBS; i++) {
        res.limbs[i] = lhs.limbs[i] - rhs.limbs[i] - borrow;
        if (lhs.limbs[i] < (rhs.limbs[i] + borrow)) {
            res.limbs[i] = res.limbs[i] + GEO_TWO_POW_WORD_SIZE;
            borrow = 1;
        } else {
            borrow = 0;
        }
    }
    return res;
}

bool geo_bigint_gte(GeoBigInt lhs, GeoBigInt rhs) {
    for (uint idx = 0; idx < GEO_NUM_LIMBS; idx++) {
        uint i = GEO_NUM_LIMBS - 1 - idx;
        if (lhs.limbs[i] < rhs.limbs[i]) return false;
        else if (lhs.limbs[i] > rhs.limbs[i]) return true;
    }
    return true;
}

bool geo_bigint_wide_gte(GeoBigIntWide lhs, GeoBigIntWide rhs) {
    for (uint idx = 0; idx < GEO_NUM_LIMBS_WIDE; idx++) {
        uint i = GEO_NUM_LIMBS_WIDE - 1 - idx;
        if (lhs.limbs[i] < rhs.limbs[i]) return false;
        else if (lhs.limbs[i] > rhs.limbs[i]) return true;
    }
    return true;
}

// --- msl-secp256k1 ff (verbatim) ---

GeoBigInt geo_ff_add(GeoBigInt a, GeoBigInt b, GeoBigInt p) {
    GeoBigIntWide p_wide;
    for (uint i = 0; i < GEO_NUM_LIMBS; i++) p_wide.limbs[i] = p.limbs[i];
    p_wide.limbs[GEO_NUM_LIMBS] = 0;
    GeoBigIntWide sum_wide = geo_bigint_add_wide(a, b);
    GeoBigInt res;
    if (geo_bigint_wide_gte(sum_wide, p_wide)) {
        GeoBigIntWide s = geo_bigint_sub_wide(sum_wide, p_wide);
        for (uint i = 0; i < GEO_NUM_LIMBS; i++) res.limbs[i] = s.limbs[i];
    } else {
        for (uint i = 0; i < GEO_NUM_LIMBS; i++) res.limbs[i] = sum_wide.limbs[i];
    }
    return res;
}

GeoBigInt geo_ff_sub(GeoBigInt a, GeoBigInt b, GeoBigInt p) {
    if (geo_bigint_gte(a, b)) {
        return geo_bigint_sub(a, b);
    } else {
        GeoBigInt r = geo_bigint_sub(b, a);
        return geo_bigint_sub(p, r);
    }
}

// --- msl-secp256k1 mont_mul_optimised (verbatim) ---

GeoBigInt geo_conditional_reduce(GeoBigInt x, GeoBigInt y) {
    if (geo_bigint_gte(x, y)) return geo_bigint_sub(x, y);
    return x;
}

GeoBigInt geo_mont_mul(GeoBigInt x, GeoBigInt y, GeoBigInt p) {
    GeoBigInt s = geo_bigint_zero();
    for (uint i = 0; i < GEO_NUM_LIMBS; i++) {
        uint t = s.limbs[0] + x.limbs[i] * y.limbs[0];
        uint tprime = t & GEO_MASK;
        uint qi = (GEO_N0 * tprime) & GEO_MASK;
        uint c = (t + qi * p.limbs[0]) >> GEO_LOG_LIMB_SIZE;
        s.limbs[0] = s.limbs[1] + x.limbs[i] * y.limbs[1] + qi * p.limbs[1] + c;
        for (uint j = 2; j < GEO_NUM_LIMBS; j++) {
            s.limbs[j-1] = s.limbs[j] + x.limbs[i] * y.limbs[j] + qi * p.limbs[j];
        }
        s.limbs[GEO_NUM_LIMBS-2] = x.limbs[i] * y.limbs[GEO_NUM_LIMBS-1] + qi * p.limbs[GEO_NUM_LIMBS-1];
    }
    uint c = 0;
    for (uint i = 0; i < GEO_NUM_LIMBS; i++) {
        uint v = s.limbs[i] + c;
        c = v >> GEO_LOG_LIMB_SIZE;
        s.limbs[i] = v & GEO_MASK;
    }
    return geo_conditional_reduce(s, p);
}

// --- msl-secp256k1 jacobian (verbatim) ---

struct GeoJacobian {
    GeoBigInt x;
    GeoBigInt y;
    GeoBigInt z;
};

GeoJacobian geo_jacobian_add(GeoJacobian a, GeoJacobian b, GeoBigInt p) {
    GeoBigInt x1=a.x, y1=a.y, z1=a.z, x2=b.x, y2=b.y, z2=b.z;

    GeoBigInt z1z1 = geo_mont_mul(z1, z1, p);
    GeoBigInt z2z2 = geo_mont_mul(z2, z2, p);
    GeoBigInt u1 = geo_mont_mul(x1, z2z2, p);
    GeoBigInt u2 = geo_mont_mul(x2, z1z1, p);
    GeoBigInt y1z2 = geo_mont_mul(y1, z2, p);
    GeoBigInt s1 = geo_mont_mul(y1z2, z2z2, p);
    GeoBigInt y2z1 = geo_mont_mul(y2, z1, p);
    GeoBigInt s2 = geo_mont_mul(y2z1, z1z1, p);
    GeoBigInt h = geo_ff_sub(u2, u1, p);
    GeoBigInt h2 = geo_ff_add(h, h, p);
    GeoBigInt i = geo_mont_mul(h2, h2, p);
    GeoBigInt j = geo_mont_mul(h, i, p);
    GeoBigInt s2s1 = geo_ff_sub(s2, s1, p);
    GeoBigInt r = geo_ff_add(s2s1, s2s1, p);
    GeoBigInt v = geo_mont_mul(u1, i, p);
    GeoBigInt v2 = geo_ff_add(v, v, p);
    GeoBigInt r2 = geo_mont_mul(r, r, p);
    GeoBigInt jv2 = geo_ff_add(j, v2, p);
    GeoBigInt x3 = geo_ff_sub(r2, jv2, p);
    GeoBigInt vx3 = geo_ff_sub(v, x3, p);
    GeoBigInt rvx3 = geo_mont_mul(r, vx3, p);
    GeoBigInt s12 = geo_ff_add(s1, s1, p);
    GeoBigInt s12j = geo_mont_mul(s12, j, p);
    GeoBigInt y3 = geo_ff_sub(rvx3, s12j, p);
    GeoBigInt z1z2 = geo_mont_mul(z1, z2, p);
    GeoBigInt z1z2h = geo_mont_mul(z1z2, h, p);
    GeoBigInt z3 = geo_ff_add(z1z2h, z1z2h, p);

    return GeoJacobian{x3, y3, z3};
}

// ============================================================
// Benchmark kernels
// ============================================================

// Their EC: 16 Jacobian additions using their exact code
kernel void bench_geo_ec_add_16(
    const device uchar* seed_be [[buffer(0)]],
    device uchar* out_be        [[buffer(1)]],
    constant uint& iterations   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // Build p in 13-bit limbs
    const uint p_data[GEO_NUM_LIMBS] = {
        7215, 8191, 8127, 8191, 8191, 8191, 8191, 8191, 8191, 8191,
        8191, 8191, 8191, 8191, 8191, 8191, 8191, 8191, 8191, 511
    };
    GeoBigInt p; for (int i=0;i<GEO_NUM_LIMBS;i++) p.limbs[i]=p_data[i];

    // Create two distinct non-zero Jacobian points from seed
    uint256 seed = uint256_from_be_device(seed_be);
    seed.d[0] ^= gid;
    GeoJacobian pt1, pt2;
    for (int i=0;i<GEO_NUM_LIMBS;i++) {
        pt1.x.limbs[i] = (seed.d[i%8] >> (i%2 ? 13 : 0)) & GEO_MASK;
        pt1.y.limbs[i] = (seed.d[(i+1)%8] >> (i%2 ? 0 : 13)) & GEO_MASK;
        pt2.x.limbs[i] = (seed.d[(i+2)%8] >> (i%2 ? 13 : 0)) & GEO_MASK;
        pt2.y.limbs[i] = (seed.d[(i+3)%8] >> (i%2 ? 0 : 13)) & GEO_MASK;
    }
    pt1.z = geo_bigint_zero(); pt1.z.limbs[0] = 1;
    pt2.z = geo_bigint_zero(); pt2.z.limbs[0] = 1;

    for (uint iter = 0; iter < iterations; iter++) {
        pt1 = geo_jacobian_add(pt1, pt2, p);
    }

    if (gid == 0) {
        // Write x coordinate to prevent dead code elimination
        uint256 r = uint256_zero();
        for (int i=0;i<8 && i<GEO_NUM_LIMBS;i++) r.d[i] = pt1.x.limbs[i];
        uint256_to_be_device(r, out_be);
    }
}

// Our EC: 16 mixed Jacobian+Affine additions using our exact code
kernel void bench_our_ec_add_16(
    const device uchar* seed_be [[buffer(0)]],
    device uchar* out_be        [[buffer(1)]],
    constant uint& iterations   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint256 seed = uint256_from_be_device(seed_be);
    seed.d[0] ^= gid;

    // Create starting Jacobian point and affine point to add
    AffinePoint ap;
    ap.x = seed;
    ap.y = seed; ap.y.d[0] ^= 0xDEADBEEF;
    JacobianPoint jp = {seed, seed, uint256_one()};
    jp.y.d[0] ^= 0xCAFEBABE;

    for (uint iter = 0; iter < iterations; iter++) {
        jp = ec_add_mixed(jp, ap);
    }

    if (gid == 0) {
        uint256_to_be_device(jp.x, out_be);
    }
}
