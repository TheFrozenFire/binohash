#include <metal_stdlib>
using namespace metal;

// ============================================================
// 256-bit unsigned integer (8 × 32-bit limbs, little-endian)
// All functions use value semantics to avoid MSL address-space issues.
// ============================================================

struct uint256 {
    uint d[8]; // d[0] = least significant
};

struct AddResult { uint256 val; uint carry; };
struct SubResult { uint256 val; uint borrow; };

uint256 uint256_zero() { return uint256{{0,0,0,0,0,0,0,0}}; }
uint256 uint256_one()  { return uint256{{1,0,0,0,0,0,0,0}}; }

uint256 uint256_from_be_thread(thread const uchar* be) {
    uint256 r;
    for (int i = 0; i < 8; i++) {
        int off = (7 - i) * 4;
        r.d[i] = ((uint)be[off]<<24) | ((uint)be[off+1]<<16) |
                 ((uint)be[off+2]<<8) | (uint)be[off+3];
    }
    return r;
}

uint256 uint256_from_be_device(const device uchar* be) {
    uint256 r;
    for (int i = 0; i < 8; i++) {
        int off = (7 - i) * 4;
        r.d[i] = ((uint)be[off]<<24) | ((uint)be[off+1]<<16) |
                 ((uint)be[off+2]<<8) | (uint)be[off+3];
    }
    return r;
}

void uint256_to_be_thread(uint256 v, thread uchar* be) {
    for (int i = 0; i < 8; i++) {
        int off = (7 - i) * 4;
        be[off]   = (uchar)(v.d[i] >> 24);
        be[off+1] = (uchar)(v.d[i] >> 16);
        be[off+2] = (uchar)(v.d[i] >> 8);
        be[off+3] = (uchar)(v.d[i]);
    }
}

void uint256_to_be_device(uint256 v, device uchar* be) {
    for (int i = 0; i < 8; i++) {
        int off = (7 - i) * 4;
        be[off]   = (uchar)(v.d[i] >> 24);
        be[off+1] = (uchar)(v.d[i] >> 16);
        be[off+2] = (uchar)(v.d[i] >> 8);
        be[off+3] = (uchar)(v.d[i]);
    }
}

bool uint256_gte(uint256 a, uint256 b) {
    for (int i = 7; i >= 0; i--) {
        if (a.d[i] > b.d[i]) return true;
        if (a.d[i] < b.d[i]) return false;
    }
    return true;
}

bool uint256_is_zero(uint256 a) {
    return (a.d[0]|a.d[1]|a.d[2]|a.d[3]|a.d[4]|a.d[5]|a.d[6]|a.d[7]) == 0;
}

AddResult uint256_add(uint256 a, uint256 b) {
    uint256 r; ulong c = 0;
    for (int i = 0; i < 8; i++) {
        c += (ulong)a.d[i] + (ulong)b.d[i];
        r.d[i] = (uint)c; c >>= 32;
    }
    return {r, (uint)c};
}

SubResult uint256_sub(uint256 a, uint256 b) {
    uint256 r; long c = 0;
    for (int i = 0; i < 8; i++) {
        c += (long)(ulong)a.d[i] - (long)(ulong)b.d[i];
        r.d[i] = (uint)c; c >>= 32;
    }
    return {r, (c < 0) ? 1u : 0u};
}

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

// ============================================================
// EC point operations (Jacobian coordinates)
// ============================================================

struct JacobianPoint { uint256 x, y, z; };
struct AffinePoint  { uint256 x, y; };

JacobianPoint ec_add_mixed(JacobianPoint p, AffinePoint q) {
    if (uint256_is_zero(p.z))
        return JacobianPoint{q.x, q.y, uint256_one()};

    uint256 z2 = field_sqr(p.z);
    uint256 u2 = field_mul(q.x, z2);
    uint256 z3 = field_mul(z2, p.z);
    uint256 s2 = field_mul(q.y, z3);
    uint256 h = field_sub(u2, p.x);
    uint256 r = field_sub(s2, p.y);

    if (uint256_is_zero(h)) {
        if (uint256_is_zero(r)) {
            uint256 a = field_sqr(p.y);
            uint256 b4 = field_mul(p.x, a);
            b4 = field_add(b4, b4); b4 = field_add(b4, b4);
            uint256 c = field_sqr(a); c = field_add(c, c);
            uint256 dd = field_sqr(p.x);
            uint256 e = field_add(dd, field_add(dd, dd));
            uint256 nx = field_sub(field_sqr(e), field_add(b4, b4));
            uint256 ny = field_sub(field_mul(e, field_sub(b4, nx)), field_add(c, c));
            uint256 nz = field_mul(p.y, p.z); nz = field_add(nz, nz);
            return JacobianPoint{nx, ny, nz};
        }
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

AffinePoint jacobian_to_affine(JacobianPoint p) {
    uint256 zi = field_inv(p.z);
    uint256 z2 = field_sqr(zi);
    uint256 z3 = field_mul(z2, zi);
    return AffinePoint{field_mul(p.x, z2), field_mul(p.y, z3)};
}

JacobianPoint ec_mul_gtable(uint256 scalar,
    const device uchar* gtx, const device uchar* gty)
{
    ushort chunks[16];
    for (int i = 0; i < 8; i++) {
        chunks[i*2]     = (ushort)(scalar.d[i] & 0xFFFF);
        chunks[i*2 + 1] = (ushort)(scalar.d[i] >> 16);
    }
    JacobianPoint result = {uint256_zero(), uint256_zero(), uint256_zero()};
    for (int chunk = 0; chunk < 16; chunk++) {
        if (chunks[chunk] == 0) continue;
        uint index = ((uint)chunk * 65536 + (uint)(chunks[chunk] - 1)) * 32;
        AffinePoint gp = {uint256_from_be_device(gtx + index),
                          uint256_from_be_device(gty + index)};
        result = ec_add_mixed(result, gp);
    }
    return result;
}

// ============================================================
// SHA-256
// ============================================================

constant uint SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

constant uint SHA256_H0[8] = {
    0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
};

inline uint rotr(uint x, uint n) { return (x >> n) | (x << (32 - n)); }

void sha256_compress(thread uint* state, thread uint* W) {
    for (int i = 16; i < 64; i++) {
        uint s0 = rotr(W[i-15],7) ^ rotr(W[i-15],18) ^ (W[i-15]>>3);
        uint s1 = rotr(W[i-2],17) ^ rotr(W[i-2],19)  ^ (W[i-2]>>10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }
    uint a=state[0],b=state[1],c=state[2],d=state[3];
    uint e=state[4],f=state[5],g=state[6],h=state[7];
    for (int i = 0; i < 64; i++) {
        uint S1 = rotr(e,6)^rotr(e,11)^rotr(e,25);
        uint ch = (e&f)^(~e&g);
        uint t1 = h + S1 + ch + SHA256_K[i] + W[i];
        uint S0 = rotr(a,2)^rotr(a,13)^rotr(a,22);
        uint maj = (a&b)^(a&c)^(b&c);
        uint t2 = S0 + maj;
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;
    state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}

void sha256_32bytes(thread const uchar* input, thread uchar* output) {
    uint W[64];
    for (int i = 0; i < 8; i++)
        W[i] = ((uint)input[i*4]<<24)|((uint)input[i*4+1]<<16)|
               ((uint)input[i*4+2]<<8)|(uint)input[i*4+3];
    W[8] = 0x80000000;
    for (int i = 9; i < 15; i++) W[i] = 0;
    W[15] = 256;
    uint state[8];
    for (int i = 0; i < 8; i++) state[i] = SHA256_H0[i];
    sha256_compress(state, W);
    for (int i = 0; i < 8; i++) {
        output[i*4]=(uchar)(state[i]>>24); output[i*4+1]=(uchar)(state[i]>>16);
        output[i*4+2]=(uchar)(state[i]>>8); output[i*4+3]=(uchar)(state[i]);
    }
}

// ============================================================
// RIPEMD-160
// ============================================================

inline uint rotl(uint x, uint n) { return (x << n) | (x >> (32 - n)); }

void ripemd160_32bytes(thread const uchar* input, thread uchar* output) {
    uint W[16];
    for (int i = 0; i < 8; i++)
        W[i] = ((uint)input[i*4]) | ((uint)input[i*4+1]<<8) |
               ((uint)input[i*4+2]<<16) | ((uint)input[i*4+3]<<24);
    W[8] = 0x00000080;
    for (int i = 9; i < 14; i++) W[i] = 0;
    W[14] = 256; W[15] = 0;

    uint h0=0x67452301,h1=0xEFCDAB89,h2=0x98BADCFE,h3=0x10325476,h4=0xC3D2E1F0;

    const uint rl[80] = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
        7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
        3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
        1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
        4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
    const uint sl[80] = {
        11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
        7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
        11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
        11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
        9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
    const uint rr[80] = {
        5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
        6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
        15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
        8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
        12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
    const uint sr[80] = {
        8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
        9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
        9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
        15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
        8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
    const uint kl[5] = {0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
    const uint kr[5] = {0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};

    uint al=h0,bl=h1,cl=h2,dl=h3,el=h4;
    uint ar=h0,br=h1,cr=h2,dr=h3,er=h4;

    for (int j = 0; j < 80; j++) {
        uint fl, fr;
        int rnd = j / 16;
        switch(rnd) {
            case 0: fl=bl^cl^dl;         fr=br^(cr|~dr);   break;
            case 1: fl=(bl&cl)|(~bl&dl);  fr=(br&dr)|(cr&~dr); break;
            case 2: fl=(bl|~cl)^dl;       fr=(br|~cr)^dr;   break;
            case 3: fl=(bl&dl)|(cl&~dl);  fr=(br&cr)|(~br&dr); break;
            default: fl=bl^(cl|~dl);      fr=br^cr^dr;      break;
        }
        uint tl = rotl(al+fl+W[rl[j]]+kl[rnd], sl[j])+el;
        al=el;el=dl;dl=rotl(cl,10);cl=bl;bl=tl;
        uint tr = rotl(ar+fr+W[rr[j]]+kr[rnd], sr[j])+er;
        ar=er;er=dr;dr=rotl(cr,10);cr=br;br=tr;
    }

    uint t = h1+cl+dr;
    h1=h2+dl+er; h2=h3+el+ar; h3=h4+al+br; h4=h0+bl+cr; h0=t;

    uint hh[5] = {h0,h1,h2,h3,h4};
    for (int i = 0; i < 5; i++) {
        output[i*4]=(uchar)(hh[i]); output[i*4+1]=(uchar)(hh[i]>>8);
        output[i*4+2]=(uchar)(hh[i]>>16); output[i*4+3]=(uchar)(hh[i]>>24);
    }
}

void hash160_33bytes(thread const uchar* input, thread uchar* output) {
    uint W[64];
    for (int i = 0; i < 8; i++)
        W[i] = ((uint)input[i*4]<<24)|((uint)input[i*4+1]<<16)|
               ((uint)input[i*4+2]<<8)|(uint)input[i*4+3];
    W[8] = ((uint)input[32]<<24) | 0x800000;
    for (int i = 9; i < 15; i++) W[i] = 0;
    W[15] = 264;
    uint state[8];
    for (int i = 0; i < 8; i++) state[i] = SHA256_H0[i];
    sha256_compress(state, W);
    uchar sha_out[32];
    for (int i = 0; i < 8; i++) {
        sha_out[i*4]=(uchar)(state[i]>>24); sha_out[i*4+1]=(uchar)(state[i]>>16);
        sha_out[i*4+2]=(uchar)(state[i]>>8); sha_out[i*4+3]=(uchar)(state[i]);
    }
    ripemd160_32bytes(sha_out, output);
}

// ============================================================
// DER check
// ============================================================

bool check_der_20(thread const uchar* d) {
    if (d[0]!=0x30) return false;
    int tl=d[1]; if (tl+3!=20) return false;
    int idx=2;
    for (int p=0;p<2;p++) {
        if (idx>=19||d[idx]!=0x02) return false; idx++;
        int il=d[idx]; idx++;
        if (il==0||idx+il>19) return false;
        if (il>1&&d[idx]==0x00&&!(d[idx+1]&0x80)) return false;
        if (d[idx]&0x80) return false; idx+=il;
    }
    return idx==19;
}

bool check_der_easy(thread const uchar* d) { return (d[0]>>4)==0x3; }

// ============================================================
// Pinning search kernel
// ============================================================

struct PinningParams {
    uint midstate[8];
    uint total_preimage_len;
    uint suffix_len;
    uint seq_offset;
    uint lt_offset;
    uint seq_value;
    uint start_lt;
    uint easy_mode;
    uint _pad;
};

kernel void pinning_search(
    constant PinningParams& params      [[buffer(0)]],
    const device uchar* suffix          [[buffer(1)]],
    const device uchar* neg_r_inv_be    [[buffer(2)]],
    const device uchar* u2r_be          [[buffer(3)]],
    const device uchar* gtable_x        [[buffer(4)]],
    const device uchar* gtable_y        [[buffer(5)]],
    device atomic_uint* hit_count       [[buffer(6)]],
    device uint* hit_indices            [[buffer(7)]],
    uint gid                            [[thread_position_in_grid]]
) {
    uint256 neg_r_inv = uint256_from_be_device(neg_r_inv_be);
    AffinePoint u2r = {uint256_from_be_device(u2r_be), uint256_from_be_device(u2r_be+32)};
    uint lt = params.start_lt + gid;

    uchar buf[128];
    for (uint i = 0; i < params.suffix_len; i++) buf[i] = suffix[i];
    buf[params.seq_offset]=(uchar)(params.seq_value);
    buf[params.seq_offset+1]=(uchar)(params.seq_value>>8);
    buf[params.seq_offset+2]=(uchar)(params.seq_value>>16);
    buf[params.seq_offset+3]=(uchar)(params.seq_value>>24);
    buf[params.lt_offset]=(uchar)(lt); buf[params.lt_offset+1]=(uchar)(lt>>8);
    buf[params.lt_offset+2]=(uchar)(lt>>16); buf[params.lt_offset+3]=(uchar)(lt>>24);

    buf[params.suffix_len]=0x80;
    for (uint i=params.suffix_len+1;i<128;i++) buf[i]=0;
    uint nblk=(params.suffix_len<56)?1:2;
    ulong bit_len=(ulong)params.total_preimage_len*8;
    uint last=nblk*64-8;
    buf[last]=(uchar)(bit_len>>56);buf[last+1]=(uchar)(bit_len>>48);
    buf[last+2]=(uchar)(bit_len>>40);buf[last+3]=(uchar)(bit_len>>32);
    buf[last+4]=(uchar)(bit_len>>24);buf[last+5]=(uchar)(bit_len>>16);
    buf[last+6]=(uchar)(bit_len>>8);buf[last+7]=(uchar)(bit_len);

    uint state[8]; for (int i=0;i<8;i++) state[i]=params.midstate[i];
    for (uint b=0;b<nblk;b++) {
        uint W[64];
        for (int i=0;i<16;i++)
            W[i]=((uint)buf[b*64+i*4]<<24)|((uint)buf[b*64+i*4+1]<<16)|
                 ((uint)buf[b*64+i*4+2]<<8)|(uint)buf[b*64+i*4+3];
        sha256_compress(state,W);
    }

    uchar first_hash[32];
    for (int i=0;i<8;i++){first_hash[i*4]=(uchar)(state[i]>>24);first_hash[i*4+1]=(uchar)(state[i]>>16);
        first_hash[i*4+2]=(uchar)(state[i]>>8);first_hash[i*4+3]=(uchar)(state[i]);}
    uchar sighash[32];
    sha256_32bytes(first_hash,sighash);

    uint256 z = uint256_from_be_thread(sighash);
    uint256 u1 = scalar_mul(neg_r_inv, z);
    JacobianPoint q = ec_mul_gtable(u1, gtable_x, gtable_y);
    q = ec_add_mixed(q, u2r);
    AffinePoint qa = jacobian_to_affine(q);

    uchar compressed[33];
    compressed[0] = (qa.y.d[0]&1) ? 0x03 : 0x02;
    uint256_to_be_thread(qa.x, compressed+1);

    uchar h160[20];
    hash160_33bytes(compressed, h160);

    bool valid = params.easy_mode ? check_der_easy(h160) : check_der_20(h160);
    if (valid) {
        uint pos = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
        if (pos < 1024) hit_indices[pos] = gid;
    }
}

// ============================================================
// Test kernels
// ============================================================

kernel void test_sha256(
    const device uchar* input [[buffer(0)]],
    device uchar* output      [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    uchar inp[32]; for (int i=0;i<32;i++) inp[i]=input[i];
    uchar out[32]; sha256_32bytes(inp, out);
    for (int i=0;i<32;i++) output[i]=out[i];
}

kernel void test_hash160(
    const device uchar* input [[buffer(0)]],
    device uchar* output      [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    uchar inp[33]; for (int i=0;i<33;i++) inp[i]=input[i];
    uchar out[20]; hash160_33bytes(inp, out);
    for (int i=0;i<20;i++) output[i]=out[i];
}

kernel void test_field_mul(
    const device uchar* a_be    [[buffer(0)]],
    const device uchar* b_be    [[buffer(1)]],
    device uchar* result_be     [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    uint256 a = uint256_from_be_device(a_be);
    uint256 b = uint256_from_be_device(b_be);
    uint256 r = field_mul(a, b);
    uint256_to_be_device(r, result_be);
}

kernel void test_field_inv(
    const device uchar* a_be    [[buffer(0)]],
    device uchar* inv_be        [[buffer(1)]],
    device uchar* product_be    [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    uint256 a = uint256_from_be_device(a_be);
    uint256 inv = field_inv(a);
    uint256 product = field_mul(a, inv);
    uint256_to_be_device(inv, inv_be);
    uint256_to_be_device(product, product_be);
}

kernel void test_ec_mul(
    const device uchar* scalar_be [[buffer(0)]],
    const device uchar* gtable_x  [[buffer(1)]],
    const device uchar* gtable_y  [[buffer(2)]],
    device uchar* out_x_be        [[buffer(3)]],
    device uchar* out_y_be        [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    uint256 scalar = uint256_from_be_device(scalar_be);
    JacobianPoint jp = ec_mul_gtable(scalar, gtable_x, gtable_y);
    AffinePoint ap = jacobian_to_affine(jp);
    uint256_to_be_device(ap.x, out_x_be);
    uint256_to_be_device(ap.y, out_y_be);
}

kernel void test_ec_recovery(
    const device uchar* digest_be    [[buffer(0)]],
    const device uchar* neg_r_inv_be [[buffer(1)]],
    const device uchar* u2r_be       [[buffer(2)]],
    const device uchar* gtable_x     [[buffer(3)]],
    const device uchar* gtable_y     [[buffer(4)]],
    device uchar* out_pubkey         [[buffer(5)]],
    device uchar* out_hash160        [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    uint256 z = uint256_from_be_device(digest_be);
    uint256 nri = uint256_from_be_device(neg_r_inv_be);
    AffinePoint u2r = {uint256_from_be_device(u2r_be), uint256_from_be_device(u2r_be+32)};

    uint256 u1 = scalar_mul(nri, z);
    JacobianPoint q = ec_mul_gtable(u1, gtable_x, gtable_y);
    q = ec_add_mixed(q, u2r);
    AffinePoint qa = jacobian_to_affine(q);

    uchar compressed[33];
    compressed[0] = (qa.y.d[0]&1) ? 0x03 : 0x02;
    uint256_to_be_thread(qa.x, compressed+1);
    for (int i=0;i<33;i++) out_pubkey[i]=compressed[i];

    uchar h160[20]; hash160_33bytes(compressed, h160);
    for (int i=0;i<20;i++) out_hash160[i]=h160[i];
}
