#include <metal_stdlib>
using namespace metal;

// ============================================================
// 256-bit unsigned integer (8 × 32-bit limbs, little-endian)
// ============================================================

struct uint256 {
    uint d[8]; // d[0] = least significant
};

uint256 uint256_zero() { return uint256{{0,0,0,0,0,0,0,0}}; }

uint256 uint256_from_be(const device uchar* be) {
    uint256 r;
    for (int i = 0; i < 8; i++) {
        int offset = (7 - i) * 4;
        r.d[i] = ((uint)be[offset] << 24) | ((uint)be[offset+1] << 16) |
                 ((uint)be[offset+2] << 8) | (uint)be[offset+3];
    }
    return r;
}

void uint256_to_be(thread const uint256& v, thread uchar* be) {
    for (int i = 0; i < 8; i++) {
        int offset = (7 - i) * 4;
        be[offset]   = (uchar)(v.d[i] >> 24);
        be[offset+1] = (uchar)(v.d[i] >> 16);
        be[offset+2] = (uchar)(v.d[i] >> 8);
        be[offset+3] = (uchar)(v.d[i]);
    }
}

bool uint256_gte(thread const uint256& a, thread const uint256& b) {
    for (int i = 7; i >= 0; i--) {
        if (a.d[i] > b.d[i]) return true;
        if (a.d[i] < b.d[i]) return false;
    }
    return true; // equal
}

bool uint256_is_zero(thread const uint256& a) {
    return (a.d[0] | a.d[1] | a.d[2] | a.d[3] | a.d[4] | a.d[5] | a.d[6] | a.d[7]) == 0;
}

// Add with carry, return carry
uint256 uint256_add(thread const uint256& a, thread const uint256& b, thread uint& carry) {
    uint256 r;
    ulong c = 0;
    for (int i = 0; i < 8; i++) {
        c += (ulong)a.d[i] + (ulong)b.d[i];
        r.d[i] = (uint)c;
        c >>= 32;
    }
    carry = (uint)c;
    return r;
}

// Subtract with borrow, return borrow
uint256 uint256_sub(thread const uint256& a, thread const uint256& b, thread uint& borrow) {
    uint256 r;
    long c = 0;
    for (int i = 0; i < 8; i++) {
        c += (long)(ulong)a.d[i] - (long)(ulong)b.d[i];
        r.d[i] = (uint)c;
        c >>= 32;
    }
    borrow = (c < 0) ? 1 : 0;
    return r;
}

// ============================================================
// secp256k1 field arithmetic (mod P)
// P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// ============================================================

constant uint256 SECP256K1_P = {{
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
}};

// Modular addition: (a + b) mod P
uint256 field_add(thread const uint256& a, thread const uint256& b) {
    uint carry;
    uint256 r = uint256_add(a, b, carry);
    if (carry || uint256_gte(r, SECP256K1_P)) {
        uint borrow;
        r = uint256_sub(r, SECP256K1_P, borrow);
    }
    return r;
}

// Modular subtraction: (a - b) mod P
uint256 field_sub(thread const uint256& a, thread const uint256& b) {
    uint borrow;
    uint256 r = uint256_sub(a, b, borrow);
    if (borrow) {
        uint carry;
        r = uint256_add(r, SECP256K1_P, carry);
    }
    return r;
}

// 256×256 → 512 multiplication (schoolbook)
void mul256_full(thread const uint256& a, thread const uint256& b, thread uint* prod) {
    for (int i = 0; i < 16; i++) prod[i] = 0;
    for (int i = 0; i < 8; i++) {
        ulong carry = 0;
        for (int j = 0; j < 8; j++) {
            carry += (ulong)prod[i+j] + (ulong)a.d[i] * (ulong)b.d[j];
            prod[i+j] = (uint)carry;
            carry >>= 32;
        }
        prod[i+8] += (uint)carry;
    }
}

// Reduce a 512-bit product mod P using secp256k1's special structure:
// P = 2^256 - 0x1000003D1, so 2^256 ≡ 0x1000003D1 (mod P)
uint256 field_reduce(thread uint* prod) {
    // Split: low = prod[0..8], high = prod[8..16]
    // result = low + high * 0x1000003D1 (mod P)
    // We may need to iterate since the addition can overflow
    uint256 r;
    for (int i = 0; i < 8; i++) r.d[i] = prod[i];

    // Multiply high part by the constant c = 0x1000003D1
    // c fits in 33 bits: 0x1_000003D1
    ulong c_lo = 0x000003D1;
    ulong c_hi = 0x1; // the 2^32 part

    for (int pass = 0; pass < 2; pass++) {
        uint high[8];
        for (int i = 0; i < 8; i++) high[i] = prod[8+i];

        // Check if high part is zero
        bool all_zero = true;
        for (int i = 0; i < 8; i++) { if (high[i]) { all_zero = false; break; } }
        if (all_zero) break;

        // Compute high * c_lo
        ulong carry = 0;
        uint tmp[9];
        for (int i = 0; i < 8; i++) {
            carry += (ulong)high[i] * c_lo;
            tmp[i] = (uint)carry;
            carry >>= 32;
        }
        tmp[8] = (uint)carry;

        // Add high * c_hi (shifted left by 32 bits, i.e., high[i] goes to position i+1)
        carry = 0;
        for (int i = 0; i < 8; i++) {
            carry += (ulong)tmp[i+1] + (ulong)high[i]; // c_hi = 1, so just add
            tmp[i+1] = (uint)carry;
            carry >>= 32;
        }

        // Add tmp[0..8] to r
        carry = 0;
        for (int i = 0; i < 8; i++) {
            carry += (ulong)r.d[i] + (ulong)tmp[i];
            r.d[i] = (uint)carry;
            carry >>= 32;
        }

        // The overflow (carry + tmp[8]) becomes the new "high" for the next pass
        ulong overflow = carry + (ulong)tmp[8];
        for (int i = 0; i < 8; i++) prod[8+i] = 0;
        prod[8] = (uint)(overflow & 0xFFFFFFFF);
        prod[9] = (uint)(overflow >> 32);
    }

    // Final reduction: if r >= P, subtract P
    if (uint256_gte(r, SECP256K1_P)) {
        uint borrow;
        r = uint256_sub(r, SECP256K1_P, borrow);
    }
    return r;
}

// Modular multiplication: (a * b) mod P
uint256 field_mul(thread const uint256& a, thread const uint256& b) {
    uint prod[16];
    mul256_full(a, b, prod);
    return field_reduce(prod);
}

// Modular squaring: a^2 mod P
uint256 field_sqr(thread const uint256& a) {
    return field_mul(a, a);
}

// Modular inverse via Fermat's little theorem: a^(P-2) mod P
// Uses a specialized addition chain for the exponent P-2
uint256 field_inv(thread const uint256& a) {
    // P - 2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    // Exponentiation using repeated squaring
    // We use the chain from libsecp256k1's implementation
    uint256 x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t1;

    x2 = field_sqr(a);
    x2 = field_mul(x2, a);          // a^3

    x3 = field_sqr(x2);
    x3 = field_mul(x3, a);          // a^7... actually let me use a simpler chain

    // Simple but correct: square-and-multiply for all 256 bits of P-2
    // P-2 in binary is all 1s except bits 1, 4, 6, 7 are 0 in the low byte (0x2D = 0b00101101)
    // For correctness over speed, use a straightforward approach

    // First compute a^(2^k) for various k, then combine
    // a^1 = a
    uint256 a1 = a;
    uint256 a2 = field_mul(a1, field_sqr(a1));  // a^3
    // Actually this is getting complex. Use the naive approach:
    // Square 256 times, multiply when the corresponding bit of P-2 is set.

    // P-2 as big-endian bytes:
    // FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2D
    constant uchar exp[32] = {
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFE, 0xFF,0xFF,0xFC,0x2D
    };

    uint256 result = {{1, 0, 0, 0, 0, 0, 0, 0}}; // 1
    uint256 base = a;

    // Process from MSB to LSB
    for (int byte_idx = 0; byte_idx < 32; byte_idx++) {
        uchar b = exp[byte_idx];
        for (int bit = 7; bit >= 0; bit--) {
            result = field_sqr(result);
            if ((b >> bit) & 1) {
                result = field_mul(result, a);
            }
        }
    }

    return result;
}

// ============================================================
// secp256k1 scalar arithmetic (mod N)
// N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// ============================================================

constant uint256 SECP256K1_N = {{
    0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
    0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
}};

// Scalar multiplication mod N (used for u1 = -r_inv * z)
// Simple schoolbook mul + Barrett-like reduction
uint256 scalar_mul(thread const uint256& a, thread const uint256& b) {
    uint prod[16];
    mul256_full(a, b, prod);

    // Reduce mod N using repeated subtraction (simple, correct)
    // For a proper implementation we'd use Barrett reduction,
    // but since this runs once per candidate it's not the bottleneck.
    uint256 r;
    for (int i = 0; i < 8; i++) r.d[i] = prod[i];

    // Add high * (2^256 - N) iteratively
    // 2^256 - N = 0x14551231950B75FC4402DA1732FC9BEBF
    // This is 129 bits, fits in 5 uint32s
    constant uint correction[5] = {0x2FC9BEBF, 0x402DA173, 0x50B75FC4, 0x14551231, 0x1};

    for (int pass = 0; pass < 2; pass++) {
        bool high_zero = true;
        for (int i = 8; i < 16; i++) { if (prod[i]) { high_zero = false; break; } }
        if (high_zero) break;

        // Multiply high[0..8] by correction[0..5] and add to low
        uint temp[16];
        for (int i = 0; i < 16; i++) temp[i] = 0;
        for (int i = 0; i < 8; i++) {
            ulong carry = 0;
            for (int j = 0; j < 5; j++) {
                carry += (ulong)temp[i+j] + (ulong)prod[8+i] * (ulong)correction[j];
                temp[i+j] = (uint)carry;
                carry >>= 32;
            }
            for (int k = i + 5; carry && k < 16; k++) {
                carry += (ulong)temp[k];
                temp[k] = (uint)carry;
                carry >>= 32;
            }
        }

        // Add temp to r, new high goes to prod[8..16]
        ulong carry = 0;
        for (int i = 0; i < 8; i++) {
            carry += (ulong)r.d[i] + (ulong)temp[i];
            r.d[i] = (uint)carry;
            carry >>= 32;
        }
        for (int i = 0; i < 8; i++) {
            prod[8+i] = temp[8+i] + ((i == 0) ? (uint)carry : 0);
        }
    }

    // Final: subtract N while r >= N
    for (int i = 0; i < 3; i++) {
        if (uint256_gte(r, SECP256K1_N)) {
            uint borrow;
            r = uint256_sub(r, SECP256K1_N, borrow);
        }
    }

    return r;
}

// ============================================================
// EC point operations (Jacobian coordinates)
// Point: (X, Y, Z) represents affine (X/Z^2, Y/Z^3)
// ============================================================

struct JacobianPoint {
    uint256 x, y, z;
};

struct AffinePoint {
    uint256 x, y;
};

// Point addition: R = P + Q (P in Jacobian, Q in affine)
// Mixed addition is cheaper than full Jacobian+Jacobian
JacobianPoint ec_add_mixed(thread const JacobianPoint& p, thread const AffinePoint& q) {
    if (uint256_is_zero(p.z)) {
        return JacobianPoint{q.x, q.y, {{1,0,0,0,0,0,0,0}}};
    }

    uint256 z2 = field_sqr(p.z);
    uint256 u2 = field_mul(q.x, z2);
    uint256 z3 = field_mul(z2, p.z);
    uint256 s2 = field_mul(q.y, z3);
    uint256 h = field_sub(u2, p.x);
    uint256 r = field_sub(s2, p.y);

    if (uint256_is_zero(h)) {
        if (uint256_is_zero(r)) {
            // Point doubling case
            uint256 a = field_sqr(p.y);
            uint256 b = field_mul(p.x, a);
            uint256 b4 = field_add(b, b); b4 = field_add(b4, b4);
            uint256 c = field_sqr(a); c = field_add(c, c);
            uint256 d = field_sqr(p.x);
            uint256 e = field_add(d, field_add(d, d)); // 3*x^2
            uint256 nx = field_sub(field_sqr(e), field_add(b4, b4));
            uint256 ny = field_sub(field_mul(e, field_sub(b4, nx)), field_add(c, c));
            uint256 nz = field_mul(p.y, p.z); nz = field_add(nz, nz);
            return JacobianPoint{nx, ny, nz};
        }
        // P = -Q, result is point at infinity
        return JacobianPoint{uint256_zero(), uint256_zero(), uint256_zero()};
    }

    uint256 h2 = field_sqr(h);
    uint256 h3 = field_mul(h2, h);
    uint256 u1h2 = field_mul(p.x, h2);

    uint256 nx = field_sub(field_sub(field_sqr(r), h3), field_add(u1h2, u1h2));
    uint256 ny = field_sub(field_mul(r, field_sub(u1h2, nx)), field_mul(p.y, h3));
    uint256 nz = field_mul(h, p.z);

    return JacobianPoint{nx, ny, nz};
}

// Convert Jacobian to Affine (requires field inversion)
AffinePoint jacobian_to_affine(thread const JacobianPoint& p) {
    uint256 z_inv = field_inv(p.z);
    uint256 z2 = field_sqr(z_inv);
    uint256 z3 = field_mul(z2, z_inv);
    return AffinePoint{field_mul(p.x, z2), field_mul(p.y, z3)};
}

// Scalar multiplication using GTable: Q = scalar * G
// GTable layout: 16 chunks × 65536 entries × 32 bytes per coordinate
// GTable[chunk][value] = value * 2^(16*chunk) * G in affine coordinates (LE bytes)
JacobianPoint ec_mul_gtable(
    thread const uint256& scalar,
    const device uchar* gtable_x,
    const device uchar* gtable_y
) {
    // Decompose scalar into 16 × 16-bit chunks
    ushort chunks[16];
    for (int i = 0; i < 8; i++) {
        chunks[i*2]     = (ushort)(scalar.d[i] & 0xFFFF);
        chunks[i*2 + 1] = (ushort)(scalar.d[i] >> 16);
    }

    JacobianPoint result = {uint256_zero(), uint256_zero(), uint256_zero()};

    for (int chunk = 0; chunk < 16; chunk++) {
        if (chunks[chunk] == 0) continue;

        uint index = ((uint)chunk * 65536 + (uint)(chunks[chunk] - 1)) * 32;
        AffinePoint gp;
        gp.x = uint256_from_be(gtable_x + index);
        gp.y = uint256_from_be(gtable_y + index);

        result = ec_add_mixed(result, gp);
    }

    return result;
}

// ============================================================
// SHA-256
// ============================================================

constant uint SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

constant uint SHA256_H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

inline uint rotr(uint x, uint n) { return (x >> n) | (x << (32 - n)); }

void sha256_compress(thread uint* state, thread uint* W) {
    for (int i = 16; i < 64; i++) {
        uint s0 = rotr(W[i-15], 7) ^ rotr(W[i-15], 18) ^ (W[i-15] >> 3);
        uint s1 = rotr(W[i-2], 17) ^ rotr(W[i-2], 19)  ^ (W[i-2] >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }
    uint a=state[0], b=state[1], c=state[2], d=state[3];
    uint e=state[4], f=state[5], g=state[6], h=state[7];
    for (int i = 0; i < 64; i++) {
        uint S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        uint ch = (e & f) ^ (~e & g);
        uint t1 = h + S1 + ch + SHA256_K[i] + W[i];
        uint S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        uint maj = (a & b) ^ (a & c) ^ (b & c);
        uint t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

// SHA-256 of a 32-byte input (single block with padding)
void sha256_32bytes(thread const uchar* input, thread uchar* output) {
    uint W[64];
    for (int i = 0; i < 8; i++)
        W[i] = ((uint)input[i*4]<<24) | ((uint)input[i*4+1]<<16) |
               ((uint)input[i*4+2]<<8) | (uint)input[i*4+3];
    W[8] = 0x80000000;
    for (int i = 9; i < 15; i++) W[i] = 0;
    W[15] = 256; // bit length

    uint state[8];
    for (int i = 0; i < 8; i++) state[i] = SHA256_H0[i];
    sha256_compress(state, W);

    for (int i = 0; i < 8; i++) {
        output[i*4]   = (uchar)(state[i] >> 24);
        output[i*4+1] = (uchar)(state[i] >> 16);
        output[i*4+2] = (uchar)(state[i] >> 8);
        output[i*4+3] = (uchar)(state[i]);
    }
}

// ============================================================
// RIPEMD-160
// ============================================================

inline uint ripemd_f(uint x, uint y, uint z) { return x ^ y ^ z; }
inline uint ripemd_g(uint x, uint y, uint z) { return (x & y) | (~x & z); }
inline uint ripemd_h(uint x, uint y, uint z) { return (x | ~y) ^ z; }
inline uint ripemd_i(uint x, uint y, uint z) { return (x & z) | (y & ~z); }
inline uint ripemd_j(uint x, uint y, uint z) { return x ^ (y | ~z); }
inline uint rotl(uint x, uint n) { return (x << n) | (x >> (32 - n)); }

void ripemd160_32bytes(thread const uchar* input, thread uchar* output) {
    // RIPEMD-160 of 32-byte input (single 64-byte block)
    uint W[16];
    for (int i = 0; i < 8; i++)
        W[i] = ((uint)input[i*4]) | ((uint)input[i*4+1]<<8) |
               ((uint)input[i*4+2]<<16) | ((uint)input[i*4+3]<<24); // LE
    W[8] = 0x00000080; // padding
    for (int i = 9; i < 14; i++) W[i] = 0;
    W[14] = 256; // bit length LE
    W[15] = 0;

    // Initial hash values
    uint h0=0x67452301, h1=0xEFCDAB89, h2=0x98BADCFE, h3=0x10325476, h4=0xC3D2E1F0;

    // Left rounds
    constant uint rl[80] = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
        7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
        3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
        1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
        4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
    };
    constant uint sl[80] = {
        11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
        7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
        11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
        11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
        9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
    };
    // Right rounds
    constant uint rr[80] = {
        5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
        6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
        15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
        8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
        12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
    };
    constant uint sr[80] = {
        8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
        9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
        9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
        15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
        8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
    };
    constant uint kl[5] = {0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
    constant uint kr[5] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000};

    uint al=h0, bl=h1, cl=h2, dl=h3, el=h4;
    uint ar=h0, br=h1, cr=h2, dr=h3, er=h4;

    for (int j = 0; j < 80; j++) {
        uint fl, fr;
        int round = j / 16;
        switch (round) {
            case 0: fl=ripemd_f(bl,cl,dl); fr=ripemd_j(br,cr,dr); break;
            case 1: fl=ripemd_g(bl,cl,dl); fr=ripemd_i(br,cr,dr); break;
            case 2: fl=ripemd_h(bl,cl,dl); fr=ripemd_h(br,cr,dr); break;
            case 3: fl=ripemd_i(bl,cl,dl); fr=ripemd_g(br,cr,dr); break;
            default: fl=ripemd_j(bl,cl,dl); fr=ripemd_f(br,cr,dr); break;
        }
        uint tl = rotl(al + fl + W[rl[j]] + kl[round], sl[j]) + el;
        al=el; el=dl; dl=rotl(cl,10); cl=bl; bl=tl;

        uint tr = rotl(ar + fr + W[rr[j]] + kr[round], sr[j]) + er;
        ar=er; er=dr; dr=rotl(cr,10); cr=br; br=tr;
    }

    uint t = h1 + cl + dr;
    h1 = h2 + dl + er;
    h2 = h3 + el + ar;
    h3 = h4 + al + br;
    h4 = h0 + bl + cr;
    h0 = t;

    // Output in LE bytes
    for (int i = 0; i < 5; i++) {
        uint v;
        switch(i) {
            case 0: v=h0; break; case 1: v=h1; break;
            case 2: v=h2; break; case 3: v=h3; break;
            default: v=h4; break;
        }
        output[i*4]   = (uchar)(v);
        output[i*4+1] = (uchar)(v >> 8);
        output[i*4+2] = (uchar)(v >> 16);
        output[i*4+3] = (uchar)(v >> 24);
    }
}

// HASH160 = RIPEMD160(SHA256(data)) for 33-byte compressed pubkey
void hash160_33bytes(thread const uchar* input, thread uchar* output) {
    // SHA-256 of 33 bytes (single block)
    uint W[64];
    for (int i = 0; i < 8; i++)
        W[i] = ((uint)input[i*4]<<24) | ((uint)input[i*4+1]<<16) |
               ((uint)input[i*4+2]<<8) | (uint)input[i*4+3];
    W[8] = ((uint)input[32]<<24) | 0x800000; // last byte + padding
    for (int i = 9; i < 15; i++) W[i] = 0;
    W[15] = 264; // 33 * 8 bits

    uint state[8];
    for (int i = 0; i < 8; i++) state[i] = SHA256_H0[i];
    sha256_compress(state, W);

    uchar sha_out[32];
    for (int i = 0; i < 8; i++) {
        sha_out[i*4]   = (uchar)(state[i] >> 24);
        sha_out[i*4+1] = (uchar)(state[i] >> 16);
        sha_out[i*4+2] = (uchar)(state[i] >> 8);
        sha_out[i*4+3] = (uchar)(state[i]);
    }

    ripemd160_32bytes(sha_out, output);
}

// ============================================================
// DER check for 20-byte RIPEMD-160 output
// ============================================================

bool check_der_20(thread const uchar* d) {
    if (d[0] != 0x30) return false;
    int tl = d[1];
    if (tl + 3 != 20) return false; // total_len + tag + len + sighash = 20
    int idx = 2;
    for (int p = 0; p < 2; p++) {
        if (idx >= 19 || d[idx] != 0x02) return false;
        idx++;
        int il = d[idx]; idx++;
        if (il == 0 || idx + il > 19) return false;
        if (il > 1 && d[idx] == 0x00 && !(d[idx+1] & 0x80)) return false;
        if (d[idx] & 0x80) return false;
        idx += il;
    }
    return idx == 19;
}

bool check_der_easy(thread const uchar* d) {
    return (d[0] >> 4) == 0x3;
}

// ============================================================
// Pinning search kernel
//
// Each thread tests one (sequence, locktime) candidate:
// 1. Finalize SHA-256d from midstate (midstate was computed over the fixed prefix)
// 2. Compute u1 = neg_r_inv * z mod N
// 3. Q = u1*G + u2R (EC point multiplication via GTable + addition)
// 4. Compress Q and compute HASH160
// 5. DER check
// ============================================================

struct PinningParams {
    uint midstate[8];       // SHA-256 state after fixed prefix blocks
    uint total_preimage_len;// total byte length of sighash preimage
    uint suffix_len;        // bytes in suffix template
    uint seq_offset;        // offset of sequence within suffix
    uint lt_offset;         // offset of locktime within suffix
    uint seq_value;         // fixed sequence value
    uint start_lt;          // starting locktime for this batch
    uint easy_mode;         // 1 = easy DER predicate, 0 = strict
};

kernel void pinning_search(
    constant PinningParams& params      [[buffer(0)]],
    const device uchar* suffix          [[buffer(1)]],   // suffix template bytes
    const device uchar* neg_r_inv_be    [[buffer(2)]],   // -r^{-1} mod N, 32 BE bytes
    const device uchar* u2r_be          [[buffer(3)]],   // u2*R: 32 BE bytes x, 32 BE bytes y
    const device uchar* gtable_x        [[buffer(4)]],   // GTable X coordinates
    const device uchar* gtable_y        [[buffer(5)]],   // GTable Y coordinates
    device atomic_uint* hit_count       [[buffer(6)]],   // number of hits found
    device uint* hit_indices            [[buffer(7)]],   // indices of hits
    uint gid                            [[thread_position_in_grid]]
) {
    // Convert BE byte buffers to uint256 (LE limbs)
    uint256 neg_r_inv = uint256_from_be(neg_r_inv_be);
    AffinePoint u2r = { uint256_from_be(u2r_be), uint256_from_be(u2r_be + 32) };

    uint lt = params.start_lt + gid;

    // Copy suffix, patch sequence and locktime
    uchar buf[128];
    for (uint i = 0; i < params.suffix_len; i++) buf[i] = suffix[i];
    buf[params.seq_offset]   = (uchar)(params.seq_value);
    buf[params.seq_offset+1] = (uchar)(params.seq_value >> 8);
    buf[params.seq_offset+2] = (uchar)(params.seq_value >> 16);
    buf[params.seq_offset+3] = (uchar)(params.seq_value >> 24);
    buf[params.lt_offset]   = (uchar)(lt);
    buf[params.lt_offset+1] = (uchar)(lt >> 8);
    buf[params.lt_offset+2] = (uchar)(lt >> 16);
    buf[params.lt_offset+3] = (uchar)(lt >> 24);

    // SHA-256 padding
    buf[params.suffix_len] = 0x80;
    for (uint i = params.suffix_len + 1; i < 128; i++) buf[i] = 0;
    uint nblk = (params.suffix_len < 56) ? 1 : 2;
    ulong bit_len = (ulong)params.total_preimage_len * 8;
    uint last = nblk * 64 - 8;
    buf[last]   = (uchar)(bit_len >> 56); buf[last+1] = (uchar)(bit_len >> 48);
    buf[last+2] = (uchar)(bit_len >> 40); buf[last+3] = (uchar)(bit_len >> 32);
    buf[last+4] = (uchar)(bit_len >> 24); buf[last+5] = (uchar)(bit_len >> 16);
    buf[last+6] = (uchar)(bit_len >> 8);  buf[last+7] = (uchar)(bit_len);

    // Finalize first SHA-256 from midstate
    uint state[8];
    for (int i = 0; i < 8; i++) state[i] = params.midstate[i];
    for (uint b = 0; b < nblk; b++) {
        uint W[64];
        for (int i = 0; i < 16; i++)
            W[i] = ((uint)buf[b*64+i*4]<<24) | ((uint)buf[b*64+i*4+1]<<16) |
                   ((uint)buf[b*64+i*4+2]<<8) | (uint)buf[b*64+i*4+3];
        sha256_compress(state, W);
    }

    // Second SHA-256 (of the 32-byte first hash)
    uchar first_hash[32];
    for (int i = 0; i < 8; i++) {
        first_hash[i*4]   = (uchar)(state[i] >> 24);
        first_hash[i*4+1] = (uchar)(state[i] >> 16);
        first_hash[i*4+2] = (uchar)(state[i] >> 8);
        first_hash[i*4+3] = (uchar)(state[i]);
    }
    uchar sighash[32];
    sha256_32bytes(first_hash, sighash);

    // Convert sighash to uint256 (big-endian bytes → LE limbs)
    uint256 z;
    for (int i = 0; i < 8; i++) {
        int off = (7-i)*4;
        z.d[i] = ((uint)sighash[off]<<24) | ((uint)sighash[off+1]<<16) |
                 ((uint)sighash[off+2]<<8) | (uint)sighash[off+3];
    }

    // u1 = neg_r_inv * z mod N
    uint256 u1 = scalar_mul(neg_r_inv, z);

    // Q = u1*G + u2*R
    JacobianPoint q = ec_mul_gtable(u1, gtable_x, gtable_y);
    q = ec_add_mixed(q, u2r);

    // Convert to affine
    AffinePoint qa = jacobian_to_affine(q);

    // Compress: 0x02/0x03 prefix + x coordinate (33 bytes)
    uchar compressed[33];
    compressed[0] = (qa.y.d[0] & 1) ? 0x03 : 0x02;
    uint256_to_be(qa.x, compressed + 1);

    // HASH160
    uchar h160[20];
    hash160_33bytes(compressed, h160);

    // DER check
    bool valid = params.easy_mode ? check_der_easy(h160) : check_der_20(h160);
    if (valid) {
        uint pos = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
        if (pos < 1024) {
            hit_indices[pos] = gid;
        }
    }
}

// ============================================================
// Test kernels — verify GPU operations against CPU reference
// ============================================================

// Test SHA-256: hash 32 input bytes, write 32 output bytes
kernel void test_sha256(
    const device uchar* input     [[buffer(0)]],
    device uchar* output          [[buffer(1)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    uchar inp[32];
    for (int i = 0; i < 32; i++) inp[i] = input[i];
    sha256_32bytes(inp, output);
}

// Test HASH160: hash 33 input bytes, write 20 output bytes
kernel void test_hash160(
    const device uchar* input     [[buffer(0)]],
    device uchar* output          [[buffer(1)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    uchar inp[33];
    for (int i = 0; i < 33; i++) inp[i] = input[i];
    hash160_33bytes(inp, output);
}

// Test field multiplication: a * b mod P, all as 32 BE bytes
kernel void test_field_mul(
    const device uchar* a_be      [[buffer(0)]],
    const device uchar* b_be      [[buffer(1)]],
    device uchar* result_be       [[buffer(2)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    uint256 a = uint256_from_be(a_be);
    uint256 b = uint256_from_be(b_be);
    uint256 r = field_mul(a, b);
    uint256_to_be(r, result_be);
}

// Test field inversion: a^(-1) mod P, verify a * a^(-1) = 1
kernel void test_field_inv(
    const device uchar* a_be      [[buffer(0)]],
    device uchar* inv_be          [[buffer(1)]],
    device uchar* product_be      [[buffer(2)]],  // a * inv(a) — should be 1
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    uint256 a = uint256_from_be(a_be);
    uint256 inv = field_inv(a);
    uint256 product = field_mul(a, inv);
    uint256_to_be(inv, inv_be);
    uint256_to_be(product, product_be);
}

// Test EC scalar multiplication: scalar * G via GTable, output affine (x, y) as BE bytes
kernel void test_ec_mul(
    const device uchar* scalar_be [[buffer(0)]],
    const device uchar* gtable_x  [[buffer(1)]],
    const device uchar* gtable_y  [[buffer(2)]],
    device uchar* out_x_be        [[buffer(3)]],
    device uchar* out_y_be        [[buffer(4)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    uint256 scalar = uint256_from_be(scalar_be);
    JacobianPoint jp = ec_mul_gtable(scalar, gtable_x, gtable_y);
    AffinePoint ap = jacobian_to_affine(jp);
    uint256_to_be(ap.x, out_x_be);
    uint256_to_be(ap.y, out_y_be);
}

// Test full EC recovery pipeline: given digest (32 BE bytes), neg_r_inv (32 BE),
// u2r (64 BE: x||y), output compressed pubkey (33 bytes) + hash160 (20 bytes)
kernel void test_ec_recovery(
    const device uchar* digest_be   [[buffer(0)]],
    const device uchar* neg_r_inv_be [[buffer(1)]],
    const device uchar* u2r_be      [[buffer(2)]],
    const device uchar* gtable_x    [[buffer(3)]],
    const device uchar* gtable_y    [[buffer(4)]],
    device uchar* out_pubkey        [[buffer(5)]],  // 33 bytes compressed
    device uchar* out_hash160       [[buffer(6)]],  // 20 bytes
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid != 0) return;

    uint256 z = uint256_from_be(digest_be);
    uint256 nri = uint256_from_be(neg_r_inv_be);
    AffinePoint u2r = { uint256_from_be(u2r_be), uint256_from_be(u2r_be + 32) };

    uint256 u1 = scalar_mul(nri, z);
    JacobianPoint q = ec_mul_gtable(u1, gtable_x, gtable_y);
    q = ec_add_mixed(q, u2r);
    AffinePoint qa = jacobian_to_affine(q);

    uchar compressed[33];
    compressed[0] = (qa.y.d[0] & 1) ? 0x03 : 0x02;
    uint256_to_be(qa.x, compressed + 1);
    for (int i = 0; i < 33; i++) out_pubkey[i] = compressed[i];

    hash160_33bytes(compressed, out_hash160);
}
