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
