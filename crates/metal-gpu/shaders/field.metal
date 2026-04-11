// ============================================================
// secp256k1 field arithmetic (mod P)
// ============================================================

uint256 secp256k1_p() {
    return uint256{{0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
                    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}};
}

uint256 field_add(uint256 a, uint256 b) {
    uint256 P = secp256k1_p();
    AddResult ar = uint256_add(a, b);
    if (ar.carry || uint256_gte(ar.val, P)) {
        SubResult sr = uint256_sub(ar.val, P);
        return sr.val;
    }
    return ar.val;
}

uint256 field_sub(uint256 a, uint256 b) {
    uint256 P = secp256k1_p();
    SubResult sr = uint256_sub(a, b);
    if (sr.borrow) {
        AddResult ar = uint256_add(sr.val, P);
        return ar.val;
    }
    return sr.val;
}

void mul256_full(uint256 a, uint256 b, thread uint* prod) {
    for (int i = 0; i < 16; i++) prod[i] = 0;
    for (int i = 0; i < 8; i++) {
        ulong carry = 0;
        for (int j = 0; j < 8; j++) {
            carry += (ulong)prod[i+j] + (ulong)a.d[i] * (ulong)b.d[j];
            prod[i+j] = (uint)carry; carry >>= 32;
        }
        prod[i+8] += (uint)carry;
    }
}

uint256 field_reduce(thread uint* prod) {
    uint256 r;
    for (int i = 0; i < 8; i++) r.d[i] = prod[i];
    uint256 P = secp256k1_p();

    // 2^256 ≡ 0x1000003D1 (mod P)
    ulong c_lo = 0x000003D1;

    for (int pass = 0; pass < 2; pass++) {
        bool all_zero = true;
        for (int i = 0; i < 8; i++) { if (prod[8+i]) { all_zero = false; break; } }
        if (all_zero) break;

        ulong carry = 0;
        uint tmp[9];
        for (int i = 0; i < 8; i++) {
            carry += (ulong)prod[8+i] * c_lo;
            tmp[i] = (uint)carry; carry >>= 32;
        }
        tmp[8] = (uint)carry;

        // Add high * 1 shifted left by 32 bits
        carry = 0;
        for (int i = 0; i < 8; i++) {
            carry += (ulong)tmp[i+1] + (ulong)prod[8+i];
            tmp[i+1] = (uint)carry; carry >>= 32;
        }

        carry = 0;
        for (int i = 0; i < 8; i++) {
            carry += (ulong)r.d[i] + (ulong)tmp[i];
            r.d[i] = (uint)carry; carry >>= 32;
        }

        ulong overflow = carry + (ulong)tmp[8];
        for (int i = 0; i < 8; i++) prod[8+i] = 0;
        prod[8] = (uint)(overflow & 0xFFFFFFFF);
        prod[9] = (uint)(overflow >> 32);
    }

    if (uint256_gte(r, P)) {
        SubResult sr = uint256_sub(r, P);
        r = sr.val;
    }
    return r;
}

uint256 field_mul(uint256 a, uint256 b) {
    uint prod[16];
    mul256_full(a, b, prod);
    return field_reduce(prod);
}

uint256 field_sqr(uint256 a) { return field_mul(a, a); }

// Repeated squaring helper: compute a^(2^n)
uint256 field_sqr_n(uint256 a, int n) {
    for (int i = 0; i < n; i++) a = field_sqr(a);
    return a;
}

uint256 field_inv(uint256 a) {
    // Addition chain for a^(P-2) mod P, from libsecp256k1.
    // P-2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    // 256 squarings + 15 multiplications (vs naive: 256 sqr + 247 mul)
    uint256 x2 = field_mul(field_sqr(a), a);                   // a^3
    uint256 x3 = field_mul(field_sqr(x2), a);                  // a^7
    uint256 x6 = field_mul(field_sqr_n(x3, 3), x3);            // a^63
    uint256 x9 = field_mul(field_sqr_n(x6, 3), x3);            // a^511
    uint256 x11 = field_mul(field_sqr_n(x9, 2), x2);           // a^2047
    uint256 x22 = field_mul(field_sqr_n(x11, 11), x11);        // a^(2^22-1)
    uint256 x44 = field_mul(field_sqr_n(x22, 22), x22);        // a^(2^44-1)
    uint256 x88 = field_mul(field_sqr_n(x44, 44), x44);        // a^(2^88-1)
    uint256 x176 = field_mul(field_sqr_n(x88, 88), x88);       // a^(2^176-1)
    uint256 x220 = field_mul(field_sqr_n(x176, 44), x44);      // a^(2^220-1)
    uint256 x223 = field_mul(field_sqr_n(x220, 3), x3);        // a^(2^223-1)
    // Exponent tail: 2^223-1 then shift and add for ...FE FFFF FC2D
    uint256 t = field_sqr_n(x223, 23);
    t = field_mul(t, x22);      // covers the 0xFFFFF part
    t = field_sqr_n(t, 5);
    t = field_mul(t, a);        // bit
    t = field_sqr_n(t, 3);
    t = field_mul(t, x2);      // 0x...2D tail: ...101101
    t = field_sqr_n(t, 2);
    t = field_mul(t, a);
    return t;
}

