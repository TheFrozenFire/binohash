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
// Batch pinning kernel — cooperative batch inversion
//
// Same pipeline as pinning_search, but threads cooperate at the
// jacobian_to_affine step via parallel prefix batch inversion.
// Requires threadgroup size = BATCH_INV_SIZE.
// ============================================================

#define BATCH_INV_SIZE 256

kernel void pinning_search_batch(
    constant PinningParams& params      [[buffer(0)]],
    const device uchar* suffix          [[buffer(1)]],
    const device uchar* neg_r_inv_be    [[buffer(2)]],
    const device uchar* u2r_be          [[buffer(3)]],
    const device uchar* gtable_x        [[buffer(4)]],
    const device uchar* gtable_y        [[buffer(5)]],
    device atomic_uint* hit_count       [[buffer(6)]],
    device uint* hit_indices            [[buffer(7)]],
    uint gid                            [[thread_position_in_grid]],
    uint tid                            [[thread_index_in_threadgroup]]
) {
    uint256 neg_r_inv = uint256_from_be_device(neg_r_inv_be);
    AffinePoint u2r = {uint256_from_be_device(u2r_be), uint256_from_be_device(u2r_be+32)};
    uint lt = params.start_lt + gid;

    // ---- Phase 1: Independent per-thread work ----
    // SHA-256d from midstate
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

    // EC recovery
    uint256 z = uint256_from_be_thread(sighash);
    uint256 u1 = scalar_mul(neg_r_inv, z);
    JacobianPoint q = ec_mul_gtable(u1, gtable_x, gtable_y);
    q = ec_add_mixed(q, u2r);

    // ---- Phase 2: Cooperative batch inversion of Z ----
    // NOTE: Sequential batch (thread 0 does all work) is 1.47x SLOWER than per-thread
    // because idle threads waste GPU execution units at barriers.
    // Parallel tree (Blelloch scan) was 2.9x faster in isolation but produces
    // incorrect individual inverses (the exclusive-scan pattern doesn't directly
    // compute element-wise inverses — it computes prefix-product inverses).
    // For now, fall back to per-thread inversion which is correct and fast.
    uint256 z_inv = field_inv(q.z);

    // ---- Phase 3: Independent per-thread work ----
    // Convert to affine using z_inv
    uint256 z2 = field_sqr(z_inv);
    uint256 z3 = field_mul(z2, z_inv);
    uint256 ax = field_mul(q.x, z2);
    uint256 ay = field_mul(q.y, z3);

    // Compress pubkey
    uchar compressed[33];
    compressed[0] = (ay.d[0] & 1) ? 0x03 : 0x02;
    uint256_to_be_thread(ax, compressed+1);

    // HASH160
    uchar h160[20];
    hash160_33bytes(compressed, h160);

    // DER check
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

// ============================================================
// Per-thread batched pinning kernel — Montgomery batch inversion
//
// Each thread processes BATCH_N candidates sequentially, accumulating
// Jacobian Z-values, then inverts all N with a single field_inv
// + 3(N-1) field_muls (Montgomery's trick). No cross-thread sync.
// ============================================================

#define BATCH_N 8

kernel void pinning_search_batched(
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

    uint base_lt = params.start_lt + gid * BATCH_N;

    // ---- Phase 1: Compute Jacobian points for all candidates ----
    uint256 jx[BATCH_N], jy[BATCH_N], jz[BATCH_N];

    for (int k = 0; k < BATCH_N; k++) {
        uint lt = base_lt + k;

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

        jx[k] = q.x;
        jy[k] = q.y;
        jz[k] = q.z;
    }

    // ---- Phase 2: Montgomery batch inversion of Z values ----
    // Accumulate running products: prod[k] = z[0] * z[1] * ... * z[k]
    uint256 prod[BATCH_N];
    prod[0] = jz[0];
    for (int k = 1; k < BATCH_N; k++)
        prod[k] = field_mul(prod[k-1], jz[k]);

    // Single expensive inversion of the accumulated product
    uint256 inv = field_inv(prod[BATCH_N - 1]);

    // Back-multiply to recover individual inverses, reusing prod[] for storage
    for (int k = BATCH_N - 1; k > 0; k--) {
        prod[k] = field_mul(inv, prod[k-1]);  // z_inv[k]
        inv = field_mul(inv, jz[k]);
    }
    prod[0] = inv;  // z_inv[0]

    // ---- Phase 3: Convert to affine and check HASH160 ----
    for (int k = 0; k < BATCH_N; k++) {
        uint256 z2 = field_sqr(prod[k]);
        uint256 z3 = field_mul(z2, prod[k]);
        uint256 ax = field_mul(jx[k], z2);
        uint256 ay = field_mul(jy[k], z3);

        uchar compressed[33];
        compressed[0] = (ay.d[0] & 1) ? 0x03 : 0x02;
        uint256_to_be_thread(ax, compressed+1);

        uchar h160[20];
        hash160_33bytes(compressed, h160);

        bool valid = params.easy_mode ? check_der_easy(h160) : check_der_20(h160);
        if (valid) {
            uint pos = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed);
            if (pos < 1024) hit_indices[pos] = gid * BATCH_N + k;
        }
    }
}

// ============================================================
// Digest search kernel — Round 1 or Round 2
//
// Each thread processes one C(n,t) subset. The kernel reads subset indices
// from a device buffer, applies FindAndDelete to the base tail (skipping
// selected dummy sig regions), continues SHA-256 from the midstate, and
// runs the standard EC recovery + HASH160 + DER check pipeline.
// ============================================================

struct DigestParams {
    uint midstate[8];
    uint total_preimage_len;   // length AFTER FindAndDelete of selected dummies
    uint base_tail_len;         // length of base_tail (all dummies present)
    uint dummy_push_len;        // bytes per dummy sig push (constant, e.g., 10)
    uint t;                     // subset size
    uint n;                     // pool size
    uint easy_mode;
    uint _pad;
};

kernel void digest_search(
    constant DigestParams& params        [[buffer(0)]],
    const device uchar* base_tail        [[buffer(1)]],
    const device uint* dummy_offsets     [[buffer(2)]],
    const device uint* subset_indices    [[buffer(3)]],
    const device uchar* neg_r_inv_be     [[buffer(4)]],
    const device uchar* u2r_be           [[buffer(5)]],
    const device uchar* gtable_x         [[buffer(6)]],
    const device uchar* gtable_y         [[buffer(7)]],
    device atomic_uint* hit_count        [[buffer(8)]],
    device uint* hit_indices             [[buffer(9)]],
    uint gid                             [[thread_position_in_grid]]
) {
    uint256 neg_r_inv = uint256_from_be_device(neg_r_inv_be);
    AffinePoint u2r = {uint256_from_be_device(u2r_be), uint256_from_be_device(u2r_be+32)};

    // Read this candidate's subset indices (max t=16 supported)
    uint subset[16];
    for (uint i = 0; i < params.t; i++) {
        subset[i] = subset_indices[gid * params.t + i];
    }

    // Compute skip ranges for each selected dummy, then sort by offset
    // (NOT by subset index — dummies are pushed in reverse in build_round_script,
    //  so higher indices have lower offsets).
    uint skip_starts[16];
    for (uint i = 0; i < params.t; i++) {
        skip_starts[i] = dummy_offsets[subset[i]];
    }
    // Insertion sort on skip_starts (ascending by offset)
    for (uint i = 1; i < params.t; i++) {
        uint key = skip_starts[i];
        int j = (int)i - 1;
        while (j >= 0 && skip_starts[j] > key) {
            skip_starts[j+1] = skip_starts[j];
            j--;
        }
        skip_starts[j+1] = key;
    }
    uint skip_ends[16];
    for (uint i = 0; i < params.t; i++) {
        skip_ends[i] = skip_starts[i] + params.dummy_push_len;
    }

    // Stream through base_tail, skipping selected dummy regions, feeding SHA-256
    uint state[8];
    for (int i = 0; i < 8; i++) state[i] = params.midstate[i];

    uchar block[64];
    uint block_pos = 0;
    uint read_pos = 0;
    uint skip_idx = 0;

    while (read_pos < params.base_tail_len) {
        // Skip over selected dummy regions
        if (skip_idx < params.t && read_pos == skip_starts[skip_idx]) {
            read_pos = skip_ends[skip_idx];
            skip_idx++;
            continue;
        }

        block[block_pos++] = base_tail[read_pos++];

        if (block_pos == 64) {
            uint W[64];
            for (int i = 0; i < 16; i++)
                W[i] = ((uint)block[i*4]<<24) | ((uint)block[i*4+1]<<16) |
                       ((uint)block[i*4+2]<<8) | (uint)block[i*4+3];
            sha256_compress(state, W);
            block_pos = 0;
        }
    }

    ulong bit_len = (ulong)params.total_preimage_len * 8;

    // SHA-256 padding: append 0x80, zeros, then 64-bit length
    block[block_pos++] = 0x80;
    if (block_pos > 56) {
        // Finish this block with zeros, compress it, then start a new block for length
        while (block_pos < 64) block[block_pos++] = 0;
        uint W[64];
        for (int i = 0; i < 16; i++)
            W[i] = ((uint)block[i*4]<<24) | ((uint)block[i*4+1]<<16) |
                   ((uint)block[i*4+2]<<8) | (uint)block[i*4+3];
        sha256_compress(state, W);
        block_pos = 0;
    }
    // Fill with zeros up to byte 56
    while (block_pos < 56) block[block_pos++] = 0;
    // Append 64-bit length (big-endian)
    block[56] = (uchar)(bit_len >> 56);
    block[57] = (uchar)(bit_len >> 48);
    block[58] = (uchar)(bit_len >> 40);
    block[59] = (uchar)(bit_len >> 32);
    block[60] = (uchar)(bit_len >> 24);
    block[61] = (uchar)(bit_len >> 16);
    block[62] = (uchar)(bit_len >> 8);
    block[63] = (uchar)(bit_len);
    {
        uint W[64];
        for (int i = 0; i < 16; i++)
            W[i] = ((uint)block[i*4]<<24) | ((uint)block[i*4+1]<<16) |
                   ((uint)block[i*4+2]<<8) | (uint)block[i*4+3];
        sha256_compress(state, W);
    }

    // First hash done; second SHA-256 to complete sighash = sha256d
    uchar first_hash[32];
    for (int i = 0; i < 8; i++) {
        first_hash[i*4]   = (uchar)(state[i] >> 24);
        first_hash[i*4+1] = (uchar)(state[i] >> 16);
        first_hash[i*4+2] = (uchar)(state[i] >> 8);
        first_hash[i*4+3] = (uchar)(state[i]);
    }
    uchar sighash[32];
    sha256_32bytes(first_hash, sighash);

    // EC recovery (same as pinning)
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
// Digest search kernel with internal nth_combination
//
// Takes a start_index (u64 split into lo/hi u32) and computes each
// thread's subset via nth_combination(n, t, start_index + gid).
// Eliminates CPU-side subset preprocessing.
// ============================================================

struct DigestNthParams {
    uint midstate[8];
    uint total_preimage_len;
    uint base_tail_len;
    uint dummy_push_len;
    uint t;
    uint n;
    uint easy_mode;
    uint binom_stride;       // row stride of binomial coefficient table
    uint start_index_lo;
    uint start_index_hi;
    uint _pad0;
    uint _pad1;
};

kernel void digest_search_nth(
    constant DigestNthParams& params     [[buffer(0)]],
    const device uchar* base_tail        [[buffer(1)]],
    const device uint* dummy_offsets     [[buffer(2)]],
    const device ulong* binom_table      [[buffer(3)]],
    const device uchar* neg_r_inv_be     [[buffer(4)]],
    const device uchar* u2r_be           [[buffer(5)]],
    const device uchar* gtable_x         [[buffer(6)]],
    const device uchar* gtable_y         [[buffer(7)]],
    device atomic_uint* hit_count        [[buffer(8)]],
    device uint* hit_indices             [[buffer(9)]],
    uint gid                             [[thread_position_in_grid]]
) {
    uint256 neg_r_inv = uint256_from_be_device(neg_r_inv_be);
    AffinePoint u2r = {uint256_from_be_device(u2r_be), uint256_from_be_device(u2r_be+32)};

    // Compute this candidate's combinatorial index
    ulong candidate_index = ((ulong)params.start_index_hi << 32) | (ulong)params.start_index_lo;
    candidate_index += (ulong)gid;

    // nth_combination via combinatorial number system
    uint subset[16];
    {
        ulong remaining = candidate_index;
        uint next = 0;
        uint remaining_k = params.t;
        for (uint i = 0; i < params.t; i++) {
            uint c = next;
            while (true) {
                uint row = params.n - 1 - c;
                ulong count = binom_table[row * params.binom_stride + (remaining_k - 1)];
                if (remaining < count) {
                    subset[i] = c;
                    next = c + 1;
                    remaining_k--;
                    break;
                }
                remaining -= count;
                c++;
            }
        }
    }

    // Compute skip ranges, sort by offset
    uint skip_starts[16];
    for (uint i = 0; i < params.t; i++) {
        skip_starts[i] = dummy_offsets[subset[i]];
    }
    for (uint i = 1; i < params.t; i++) {
        uint key = skip_starts[i];
        int j = (int)i - 1;
        while (j >= 0 && skip_starts[j] > key) {
            skip_starts[j+1] = skip_starts[j];
            j--;
        }
        skip_starts[j+1] = key;
    }
    uint skip_ends[16];
    for (uint i = 0; i < params.t; i++) {
        skip_ends[i] = skip_starts[i] + params.dummy_push_len;
    }

    // Stream through base_tail, feeding SHA-256
    uint state[8];
    for (int i = 0; i < 8; i++) state[i] = params.midstate[i];

    uchar block[64];
    uint block_pos = 0;
    uint read_pos = 0;
    uint skip_idx = 0;

    while (read_pos < params.base_tail_len) {
        if (skip_idx < params.t && read_pos == skip_starts[skip_idx]) {
            read_pos = skip_ends[skip_idx];
            skip_idx++;
            continue;
        }
        block[block_pos++] = base_tail[read_pos++];
        if (block_pos == 64) {
            uint W[64];
            for (int i = 0; i < 16; i++)
                W[i] = ((uint)block[i*4]<<24) | ((uint)block[i*4+1]<<16) |
                       ((uint)block[i*4+2]<<8) | (uint)block[i*4+3];
            sha256_compress(state, W);
            block_pos = 0;
        }
    }

    ulong bit_len = (ulong)params.total_preimage_len * 8;
    block[block_pos++] = 0x80;
    if (block_pos > 56) {
        while (block_pos < 64) block[block_pos++] = 0;
        uint W[64];
        for (int i = 0; i < 16; i++)
            W[i] = ((uint)block[i*4]<<24) | ((uint)block[i*4+1]<<16) |
                   ((uint)block[i*4+2]<<8) | (uint)block[i*4+3];
        sha256_compress(state, W);
        block_pos = 0;
    }
    while (block_pos < 56) block[block_pos++] = 0;
    block[56] = (uchar)(bit_len >> 56);
    block[57] = (uchar)(bit_len >> 48);
    block[58] = (uchar)(bit_len >> 40);
    block[59] = (uchar)(bit_len >> 32);
    block[60] = (uchar)(bit_len >> 24);
    block[61] = (uchar)(bit_len >> 16);
    block[62] = (uchar)(bit_len >> 8);
    block[63] = (uchar)(bit_len);
    {
        uint W[64];
        for (int i = 0; i < 16; i++)
            W[i] = ((uint)block[i*4]<<24) | ((uint)block[i*4+1]<<16) |
                   ((uint)block[i*4+2]<<8) | (uint)block[i*4+3];
        sha256_compress(state, W);
    }

    uchar first_hash[32];
    for (int i = 0; i < 8; i++) {
        first_hash[i*4]   = (uchar)(state[i] >> 24);
        first_hash[i*4+1] = (uchar)(state[i] >> 16);
        first_hash[i*4+2] = (uchar)(state[i] >> 8);
        first_hash[i*4+3] = (uchar)(state[i]);
    }
    uchar sighash[32];
    sha256_32bytes(first_hash, sighash);

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

kernel void test_scalar_mul(
    const device uchar* a_be    [[buffer(0)]],
    const device uchar* b_be    [[buffer(1)]],
    device uchar* result_be     [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    uint256 a = uint256_from_be_device(a_be);
    uint256 b = uint256_from_be_device(b_be);
    uint256 r = scalar_mul(a, b);
    uint256_to_be_device(r, result_be);
}

// ============================================================
// Benchmark kernels — tight loops to measure field operation throughput
// ============================================================

// Run N iterations of field_mul, chain-dependent to prevent optimization.
// Each thread starts from a different value (based on gid) for realistic divergence.
kernel void bench_field_mul(
    const device uchar* seed_be   [[buffer(0)]],
    device uchar* out_be          [[buffer(1)]],
    constant uint& iterations     [[buffer(2)]],
    uint gid                      [[thread_position_in_grid]]
) {
    uint256 a = uint256_from_be_device(seed_be);
    // Mix in gid to give each thread a unique starting point
    a.d[0] ^= gid;
    uint256 b = field_add(a, uint256_one());

    for (uint i = 0; i < iterations; i++) {
        a = field_mul(a, b);
    }
    // Write result to prevent dead-code elimination
    if (gid == 0) uint256_to_be_device(a, out_be);
}

// Run N iterations of field_sqr (currently calls field_mul(a,a)).
kernel void bench_field_sqr(
    const device uchar* seed_be   [[buffer(0)]],
    device uchar* out_be          [[buffer(1)]],
    constant uint& iterations     [[buffer(2)]],
    uint gid                      [[thread_position_in_grid]]
) {
    uint256 a = uint256_from_be_device(seed_be);
    a.d[0] ^= gid;

    for (uint i = 0; i < iterations; i++) {
        a = field_sqr(a);
    }
    if (gid == 0) uint256_to_be_device(a, out_be);
}

// Run N iterations of field_inv (the big one — each inv does 271 field_muls).
kernel void bench_field_inv_loop(
    const device uchar* seed_be   [[buffer(0)]],
    device uchar* out_be          [[buffer(1)]],
    constant uint& iterations     [[buffer(2)]],
    uint gid                      [[thread_position_in_grid]]
) {
    uint256 a = uint256_from_be_device(seed_be);
    a.d[0] ^= gid;
    if (uint256_is_zero(a)) a = uint256_one(); // avoid inverting zero

    for (uint i = 0; i < iterations; i++) {
        a = field_inv(a);
    }
    if (gid == 0) uint256_to_be_device(a, out_be);
}
