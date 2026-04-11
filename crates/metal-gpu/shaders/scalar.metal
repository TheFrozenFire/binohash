// ============================================================
// Scalar arithmetic (mod N)
// ============================================================

uint256 secp256k1_n() {
    return uint256{{0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
                    0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}};
}

uint256 scalar_mul(uint256 a, uint256 b) {
    uint prod[16];
    mul256_full(a, b, prod);
    uint256 N = secp256k1_n();

    const uint correction[5] = {0x2FC9BEBF, 0x402DA173, 0x50B75FC4, 0x14551231, 0x1};
    uint256 r;
    for (int i = 0; i < 8; i++) r.d[i] = prod[i];

    for (int pass = 0; pass < 2; pass++) {
        bool high_zero = true;
        for (int i = 8; i < 16; i++) { if (prod[i]) { high_zero = false; break; } }
        if (high_zero) break;

        uint temp[16];
        for (int i = 0; i < 16; i++) temp[i] = 0;
        for (int i = 0; i < 8; i++) {
            ulong carry = 0;
            for (int j = 0; j < 5; j++) {
                carry += (ulong)temp[i+j] + (ulong)prod[8+i] * (ulong)correction[j];
                temp[i+j] = (uint)carry; carry >>= 32;
            }
            for (int k = i+5; carry && k < 16; k++) {
                carry += (ulong)temp[k]; temp[k] = (uint)carry; carry >>= 32;
            }
        }

        ulong carry = 0;
        for (int i = 0; i < 8; i++) {
            carry += (ulong)r.d[i] + (ulong)temp[i];
            r.d[i] = (uint)carry; carry >>= 32;
        }
        for (int i = 0; i < 8; i++) prod[8+i] = temp[8+i] + ((i==0) ? (uint)carry : 0);
    }

    for (int i = 0; i < 3; i++) {
        if (uint256_gte(r, N)) { SubResult sr = uint256_sub(r, N); r = sr.val; }
    }
    return r;
}

