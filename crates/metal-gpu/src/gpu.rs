use metal::*;
use std::mem;
use std::path::Path;

use der::ParsedDerSig;
use ecdsa_recovery::SECP256K1_N;
use script::find_and_delete;
use secp256k1::{PublicKey, Scalar, Secp256k1, SecretKey};
use tx::Transaction;

// ============================================================
// Scalar field arithmetic (mod N)
// ============================================================

/// Compute the modular inverse of a scalar: a^(-1) mod N.
///
/// Uses Fermat's little theorem: a^(-1) = a^(N-2) mod N, implemented via
/// square-and-multiply using the secp256k1 crate's scalar operations.
fn scalar_inv(a: &[u8; 32]) -> [u8; 32] {
    // N - 2
    let mut exp = SECP256K1_N;
    let mut borrow: u16 = 2;
    for i in (0..32).rev() {
        let diff = exp[i] as u16 + 256 - borrow;
        exp[i] = diff as u8;
        borrow = if diff < 256 { 1 } else { 0 };
    }

    let a_scalar = Scalar::from_be_bytes(*a).expect("valid scalar");

    // Square-and-multiply: a^(N-2) mod N
    let mut result: Option<SecretKey> = None;
    for byte in exp {
        for bit in (0..8).rev() {
            if let Some(ref mut r) = result {
                // Square
                let r_scalar = Scalar::from_be_bytes(r.secret_bytes()).expect("valid");
                *r = r.mul_tweak(&r_scalar).expect("valid");
                // Multiply if bit is set
                if (byte >> bit) & 1 == 1 {
                    *r = r.mul_tweak(&a_scalar).expect("valid");
                }
            } else if (byte >> bit) & 1 == 1 {
                result = Some(SecretKey::from_byte_array(*a).expect("valid"));
            }
        }
    }

    result.expect("N-2 is nonzero").secret_bytes()
}

/// Negate a scalar: result = N - a (mod N).
fn scalar_negate(a: &[u8; 32]) -> [u8; 32] {
    SecretKey::from_byte_array(*a)
        .expect("valid")
        .negate()
        .secret_bytes()
}

/// Multiply two scalars: result = a * b (mod N).
fn scalar_mul_mod(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let sk = SecretKey::from_byte_array(*a).expect("valid");
    let scalar = Scalar::from_be_bytes(*b).expect("valid");
    sk.mul_tweak(&scalar).expect("valid").secret_bytes()
}

// ============================================================
// GPU search parameters
// ============================================================

/// Precomputed parameters for GPU pinning search, derived from a real
/// nonce signature and transaction template.
#[derive(Debug, Clone)]
pub struct GpuSearchParams {
    /// SHA-256 state after processing the fixed prefix of the sighash preimage.
    pub midstate: [u32; 8],
    /// The variable suffix of the sighash preimage (starts at the midstate boundary).
    pub suffix: Vec<u8>,
    /// Total length of the full sighash preimage (prefix + suffix).
    pub total_preimage_len: u32,
    /// Offset of the sequence field within the suffix.
    pub seq_offset: u32,
    /// Offset of the locktime field within the suffix.
    pub lt_offset: u32,
    /// -r^(-1) mod N, as 32 big-endian bytes.
    pub neg_r_inv: [u8; 32],
    /// x-coordinate of u2*R, as 32 big-endian bytes.
    pub u2r_x: [u8; 32],
    /// y-coordinate of u2*R, as 32 big-endian bytes.
    pub u2r_y: [u8; 32],
}

/// Precomputed parameters for GPU digest search (Round 1 or Round 2).
#[derive(Debug, Clone)]
pub struct GpuDigestSearchParams {
    /// SHA-256 state after processing the fixed prefix (same for all subsets).
    pub midstate: [u32; 8],
    /// Length of the fixed prefix (in bytes), for padding calculations.
    pub prefix_len: u32,
    /// Base tail template: the bytes from midstate_boundary to end of preimage,
    /// with the round's nonce sig already FindAndDeleted but dummy sigs intact.
    pub base_tail: Vec<u8>,
    /// Offset of each dummy sig's push (including the length prefix) within base_tail.
    /// dummy_offsets.len() == n (the pool size).
    pub dummy_offsets: Vec<u32>,
    /// Length of each dummy sig push (e.g., 10 bytes for 9-byte sigs + 1 length prefix).
    pub dummy_push_len: u32,
    /// -r^(-1) mod N for the round's nonce sig.
    pub neg_r_inv: [u8; 32],
    /// u2*R x-coordinate.
    pub u2r_x: [u8; 32],
    /// u2*R y-coordinate.
    pub u2r_y: [u8; 32],
}

impl GpuDigestSearchParams {
    /// Build digest search params from a round's nonce sig, transaction, and full script.
    ///
    /// `t` is the subset size (number of selected dummy sigs per candidate).
    /// The resulting params describe the shared midstate (fixed prefix) and the
    /// template tail. For each subset, the GPU will apply FindAndDelete to remove
    /// the selected dummy sigs from the tail.
    pub fn from_digest_search(
        sig_nonce: &ParsedDerSig,
        sig_nonce_bytes: &[u8],
        dummy_sigs: &[Vec<u8>],
        tx: &Transaction,
        full_script: &[u8],
        input_index: usize,
        t: usize,
    ) -> Self {
        let secp = Secp256k1::new();

        // ---- Scalar precomputations (same as pinning) ----
        let r_inv = scalar_inv(&sig_nonce.r);
        let neg_r_inv = scalar_negate(&r_inv);
        let u2 = scalar_mul_mod(&sig_nonce.s, &r_inv);

        let mut r_compressed = [0u8; 33];
        r_compressed[0] = 0x02;
        r_compressed[1..].copy_from_slice(&sig_nonce.r);
        let r_point = PublicKey::from_slice(&r_compressed).expect("r is a valid x-coordinate");

        let u2_scalar = Scalar::from_be_bytes(u2).expect("valid scalar");
        let u2r = r_point.mul_tweak(&secp, &u2_scalar).expect("valid tweak");
        let u2r_uncompressed = u2r.serialize_uncompressed();
        let mut u2r_x = [0u8; 32];
        let mut u2r_y = [0u8; 32];
        u2r_x.copy_from_slice(&u2r_uncompressed[1..33]);
        u2r_y.copy_from_slice(&u2r_uncompressed[33..65]);

        // ---- Base script code: full script with nonce sig removed ----
        let base_script_code = find_and_delete(full_script, sig_nonce_bytes);

        // ---- Build the base preimage (no dummies removed yet) ----
        // The script_code_len varint in this preimage encodes base_script_code.len().
        // But the CPU's final preimage has a varint encoding the length AFTER dummy
        // removal. We patch the varint to reflect the final length (constant per round
        // since t is fixed).
        let dummy_push_len_usize = script::push_data(&dummy_sigs[0]).len();
        let final_script_code_len = base_script_code.len() - t * dummy_push_len_usize;

        let mut base_preimage = tx
            .legacy_sighash_preimage(input_index, &base_script_code, sig_nonce.sighash_type)
            .expect("valid preimage");

        // Patch the script_code_len varint. For our script sizes (~1.5KB to ~10KB),
        // this is a 3-byte varint: 0xfd followed by 2 bytes little-endian.
        // Varint starts at offset: 4 (version) + 1 (input_count) + 32 (txid) + 4 (vout) = 41
        let varint_offset = 41;
        assert_eq!(
            base_preimage[varint_offset], 0xfd,
            "expected 3-byte varint for script_code_len (got {:02x})",
            base_preimage[varint_offset]
        );
        assert!(
            (253..=65535).contains(&final_script_code_len),
            "final script_code_len must fit in a 3-byte varint (got {final_script_code_len})"
        );
        base_preimage[varint_offset + 1] = (final_script_code_len & 0xff) as u8;
        base_preimage[varint_offset + 2] = ((final_script_code_len >> 8) & 0xff) as u8;

        // ---- Find dummy sig positions in the base preimage ----
        // Each dummy sig is pushed with its length prefix (for 9-byte sigs, that's "09" + 9 bytes).
        // We need to find where each dummy's push appears in the base_preimage.
        let mut dummy_offsets = Vec::with_capacity(dummy_sigs.len());
        let dummy_push_len = {
            let sample_push = script::push_data(&dummy_sigs[0]);
            sample_push.len() as u32
        };

        for (di, dummy) in dummy_sigs.iter().enumerate() {
            let push = script::push_data(dummy);
            // Find ALL occurrences and take the last one (Round 2 dummies are at the
            // end of the script, so the last occurrence is the correct one).
            let mut positions: Vec<usize> = Vec::new();
            let mut start = 0;
            while let Some(p) = find_subsequence(&base_preimage[start..], &push) {
                positions.push(start + p);
                start = start + p + 1;
            }
            assert!(
                !positions.is_empty(),
                "dummy sig {di} not found in preimage"
            );
            // Use the LAST occurrence (Round 2 dummies are at the end of the script)
            let pos = *positions.last().unwrap();
            dummy_offsets.push(pos as u32);
        }

        // ---- Determine the midstate boundary ----
        // It must be at a 64-byte boundary BEFORE the lowest dummy offset.
        let min_dummy_offset = *dummy_offsets.iter().min().expect("at least one dummy");
        let midstate_boundary = ((min_dummy_offset / 64) * 64) as usize;

        let prefix = &base_preimage[..midstate_boundary];
        let base_tail = base_preimage[midstate_boundary..].to_vec();

        // Adjust dummy offsets to be relative to base_tail
        for offset in dummy_offsets.iter_mut() {
            *offset -= midstate_boundary as u32;
        }

        let midstate = hash::sha256_midstate(prefix);

        GpuDigestSearchParams {
            midstate,
            prefix_len: midstate_boundary as u32,
            base_tail,
            dummy_offsets,
            dummy_push_len,
            neg_r_inv,
            u2r_x,
            u2r_y,
        }
    }
}

/// Find the first occurrence of `needle` in `haystack`.
fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

impl GpuSearchParams {
    /// Build GPU search parameters from a pinning nonce signature, transaction
    /// template, and script code.
    ///
    /// The `script_code` should already have FindAndDelete applied for the
    /// nonce signature (this matches how the CPU search prepares it).
    pub fn from_pinning_search(
        sig_nonce: &ParsedDerSig,
        tx: &Transaction,
        full_script: &[u8],
        sig_nonce_bytes: &[u8],
        input_index: usize,
    ) -> Self {
        let secp = Secp256k1::new();

        // ---- Scalar precomputations ----
        // r_inv = r^(-1) mod N
        let r_inv = scalar_inv(&sig_nonce.r);
        // neg_r_inv = -r^(-1) mod N
        let neg_r_inv = scalar_negate(&r_inv);
        // u2 = s * r_inv mod N
        let u2 = scalar_mul_mod(&sig_nonce.s, &r_inv);

        // ---- EC point recovery ----
        // R = curve point with x = r (even y, recovery_id = 0)
        let mut r_compressed = [0u8; 33];
        r_compressed[0] = 0x02;
        r_compressed[1..].copy_from_slice(&sig_nonce.r);
        let r_point = PublicKey::from_slice(&r_compressed).expect("r is a valid x-coordinate");

        // u2r = u2 * R (EC scalar multiplication)
        let u2_scalar = Scalar::from_be_bytes(u2).expect("valid scalar");
        let u2r = r_point.mul_tweak(&secp, &u2_scalar).expect("valid tweak");
        let u2r_uncompressed = u2r.serialize_uncompressed();
        let mut u2r_x = [0u8; 32];
        let mut u2r_y = [0u8; 32];
        u2r_x.copy_from_slice(&u2r_uncompressed[1..33]);
        u2r_y.copy_from_slice(&u2r_uncompressed[33..65]);

        // ---- Sighash preimage decomposition ----
        // Script code after FindAndDelete of the nonce sig
        let pin_script_code = find_and_delete(full_script, sig_nonce_bytes);

        // Find sequence and locktime offsets by diffing two preimages
        let (seq_abs, lt_abs) = {
            let mut tx1 = tx.clone();
            tx1.inputs[input_index].sequence = 0xAAAAAAAA;
            tx1.locktime = 0xBBBBBBBB;
            let pre1 = tx1
                .legacy_sighash_preimage(input_index, &pin_script_code, sig_nonce.sighash_type)
                .expect("valid preimage");

            let mut tx2 = tx.clone();
            tx2.inputs[input_index].sequence = 0xCCCCCCCC;
            tx2.locktime = 0xDDDDDDDD;
            let pre2 = tx2
                .legacy_sighash_preimage(input_index, &pin_script_code, sig_nonce.sighash_type)
                .expect("valid preimage");

            assert_eq!(pre1.len(), pre2.len(), "preimage lengths must be identical");

            let mut seq_start = None;
            let mut lt_start = None;
            for i in 0..pre1.len() {
                if pre1[i] != pre2[i] {
                    if seq_start.is_none() {
                        seq_start = Some(i);
                    } else if lt_start.is_none() && i >= seq_start.unwrap() + 4 {
                        lt_start = Some(i);
                    }
                }
            }
            (seq_start.expect("sequence field not found"), lt_start.expect("locktime field not found"))
        };

        // Split at the last 64-byte boundary before the sequence field
        let midstate_boundary = (seq_abs / 64) * 64;

        // Build the actual preimage to extract prefix and suffix
        let preimage = tx
            .legacy_sighash_preimage(input_index, &pin_script_code, sig_nonce.sighash_type)
            .expect("valid preimage");

        let prefix = &preimage[..midstate_boundary];
        let suffix = preimage[midstate_boundary..].to_vec();

        let midstate = hash::sha256_midstate(prefix);

        GpuSearchParams {
            midstate,
            total_preimage_len: preimage.len() as u32,
            seq_offset: (seq_abs - midstate_boundary) as u32,
            lt_offset: (lt_abs - midstate_boundary) as u32,
            suffix,
            neg_r_inv,
            u2r_x,
            u2r_y,
        }
    }
}

const SHADER_SOURCE: &str = concat!(
    include_str!("../shaders/uint256.metal"),
    include_str!("../shaders/field.metal"),
    include_str!("../shaders/scalar.metal"),
    include_str!("../shaders/ec.metal"),
    include_str!("../shaders/sha256.metal"),
    include_str!("../shaders/ripemd160.metal"),
    include_str!("../shaders/der.metal"),
    include_str!("../shaders/kernels.metal"),
    include_str!("../shaders/montgomery.metal"),
    include_str!("../shaders/mont_benchmark_comparison.metal"),
    include_str!("../shaders/ec_comparison.metal"),
    include_str!("../shaders/batch_inv.metal"),
    include_str!("../shaders/hash_bench.metal"),
);
const MAX_HITS: usize = 1024;
const THREADS_PER_GROUP: u64 = 256;

/// GTable: 16 chunks × 65536 entries × 32 bytes per coordinate (X and Y separately)
const GTABLE_CHUNKS: usize = 16;
const GTABLE_ENTRIES: usize = 65536;
const GTABLE_POINT_BYTES: usize = 32;
const GTABLE_BYTES: usize = GTABLE_CHUNKS * GTABLE_ENTRIES * GTABLE_POINT_BYTES;

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("no Metal device found")]
    NoDevice,
    #[error("shader compilation failed: {0}")]
    ShaderCompilation(String),
    #[error("kernel function not found: {0}")]
    KernelNotFound(String),
    #[error("pipeline creation failed")]
    PipelineCreation,
    #[error("GTable cache load failed")]
    GTableLoad,
}

/// Precomputed values for the nonce signature, used across all candidates.
#[repr(C)]
struct PinningParamsGpu {
    midstate: [u32; 8],
    total_preimage_len: u32,
    suffix_len: u32,
    seq_offset: u32,
    lt_offset: u32,
    seq_value: u32,
    start_lt: u32,
    easy_mode: u32,
    _pad: u32, // align to 16 bytes
}

/// A hit found by the GPU pinning search.
#[derive(Debug, Clone)]
pub struct GpuPinningHit {
    /// The locktime value that produced a DER hit.
    pub locktime: u32,
    /// The thread index within the batch.
    pub thread_index: u32,
}

/// Metal GPU miner for the Binohash pinning search.
pub struct MetalMiner {
    device: Device,
    queue: CommandQueue,
    library: Library,
    pinning_pipeline: ComputePipelineState,
    // GTable buffers (persistent across batches)
    gtable_x_buf: Buffer,
    gtable_y_buf: Buffer,
}

impl MetalMiner {
    /// Create a new MetalMiner with precomputed GTable.
    ///
    /// The GTable is either loaded from the cache file or computed from scratch.
    /// First-time computation takes ~30-60 seconds.
    pub fn new(cache_path: Option<&Path>) -> Result<Self, GpuError> {
        let device = Device::system_default().ok_or(GpuError::NoDevice)?;
        let queue = device.new_command_queue();

        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_SOURCE, &options)
            .map_err(|e| GpuError::ShaderCompilation(e.to_string()))?;

        let pinning_fn = library
            .get_function("pinning_search", None)
            .map_err(|_| GpuError::KernelNotFound("pinning_search".into()))?;

        let pinning_pipeline = device
            .new_compute_pipeline_state_with_function(&pinning_fn)
            .map_err(|_| GpuError::PipelineCreation)?;

        // Allocate GTable buffers
        let gtable_x_buf = device.new_buffer(
            GTABLE_BYTES as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let gtable_y_buf = device.new_buffer(
            GTABLE_BYTES as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let mut miner = MetalMiner {
            device,
            queue,
            library,
            pinning_pipeline,
            gtable_x_buf,
            gtable_y_buf,
        };

        // Load or compute GTable
        if let Some(path) = cache_path {
            if path.exists() {
                miner.load_gtable_cache(path)?;
            } else {
                miner.compute_gtable();
                miner.save_gtable_cache(path);
            }
        } else {
            miner.compute_gtable();
        }

        Ok(miner)
    }

    /// Compute the GTable: GTable[chunk][j] = (j+1) * 2^(16*chunk) * G
    ///
    /// Uses the secp256k1 crate for point arithmetic. Each entry stores the
    /// affine x and y coordinates as 32 big-endian bytes.
    fn compute_gtable(&mut self) {
        use secp256k1::{PublicKey, Secp256k1, SecretKey};

        let secp = Secp256k1::new();
        let gt_x = self.gtable_x_buf.contents() as *mut u8;
        let gt_y = self.gtable_y_buf.contents() as *mut u8;

        // For each chunk, base = 2^(16*chunk) * G
        // GTable[chunk][j] = (j+1) * base
        let mut base_scalar = [0u8; 32];
        base_scalar[31] = 1; // start with scalar = 1

        for chunk in 0..GTABLE_CHUNKS {
            // Compute base point = base_scalar * G
            let sk = SecretKey::from_byte_array(base_scalar).expect("valid scalar");
            let base_point = PublicKey::from_secret_key(&secp, &sk);

            // Accumulate: current = base_point, then current += base_point each iteration
            let mut current = base_point;

            for j in 0..GTABLE_ENTRIES - 1 {
                let elem = chunk * GTABLE_ENTRIES + j;
                let serialized = current.serialize_uncompressed(); // 65 bytes: 0x04 || x || y

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        serialized[1..33].as_ptr(), // x (big-endian)
                        gt_x.add(elem * GTABLE_POINT_BYTES),
                        32,
                    );
                    std::ptr::copy_nonoverlapping(
                        serialized[33..65].as_ptr(), // y (big-endian)
                        gt_y.add(elem * GTABLE_POINT_BYTES),
                        32,
                    );
                }

                // current += base_point
                current = current.combine(&base_point).expect("point addition");
            }

            // Multiply base_scalar by 2^16 for next chunk using secp256k1's tweak API
            // 65536 = 0x10000 in big-endian 32 bytes
            let mut shift_bytes = [0u8; 32];
            shift_bytes[29] = 0x01; // 0x00..00_0001_0000
            let shift_scalar =
                secp256k1::Scalar::from_be_bytes(shift_bytes).expect("65536 is valid scalar");
            let tweaked = SecretKey::from_byte_array(base_scalar)
                .expect("valid scalar")
                .mul_tweak(&shift_scalar)
                .expect("tweak");
            base_scalar = tweaked.secret_bytes();
        }
    }

    fn load_gtable_cache(&mut self, path: &Path) -> Result<(), GpuError> {
        let data = std::fs::read(path).map_err(|_| GpuError::GTableLoad)?;
        if data.len() != GTABLE_BYTES * 2 {
            return Err(GpuError::GTableLoad);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.gtable_x_buf.contents() as *mut u8,
                GTABLE_BYTES,
            );
            std::ptr::copy_nonoverlapping(
                data[GTABLE_BYTES..].as_ptr(),
                self.gtable_y_buf.contents() as *mut u8,
                GTABLE_BYTES,
            );
        }
        Ok(())
    }

    fn save_gtable_cache(&self, path: &Path) {
        let mut data = vec![0u8; GTABLE_BYTES * 2];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.gtable_x_buf.contents() as *const u8,
                data.as_mut_ptr(),
                GTABLE_BYTES,
            );
            std::ptr::copy_nonoverlapping(
                self.gtable_y_buf.contents() as *const u8,
                data[GTABLE_BYTES..].as_mut_ptr(),
                GTABLE_BYTES,
            );
        }
        let _ = std::fs::write(path, &data);
    }

    /// Run a pinning search batch on the GPU.
    ///
    /// Tests `batch_size` locktime values starting from `start_lt`, with
    /// a fixed sequence value. Returns any hits found.
    ///
    /// `midstate`: SHA-256 state after processing the fixed prefix of the sighash preimage
    /// `suffix`: the variable bytes of the sighash preimage (after the midstate boundary)
    /// `sig_nonce`: the parsed nonce signature (provides r for neg_r_inv computation)
    pub fn search_pinning_batch(
        &self,
        midstate: &[u32; 8],
        suffix: &[u8],
        total_preimage_len: u32,
        seq_offset: u32,
        lt_offset: u32,
        seq_value: u32,
        start_lt: u32,
        batch_size: u32,
        neg_r_inv: &[u8; 32],
        u2r_x: &[u8; 32],
        u2r_y: &[u8; 32],
        easy_mode: bool,
    ) -> Vec<GpuPinningHit> {
        let params = PinningParamsGpu {
            midstate: *midstate,
            total_preimage_len,
            suffix_len: suffix.len() as u32,
            seq_offset,
            lt_offset,
            seq_value,
            start_lt,
            easy_mode: if easy_mode { 1 } else { 0 },
            _pad: 0,
        };

        // Create buffers
        let params_buf = self.device.new_buffer_with_data(
            &params as *const PinningParamsGpu as *const _,
            mem::size_of::<PinningParamsGpu>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let suffix_buf = self.device.new_buffer_with_data(
            suffix.as_ptr() as *const _,
            suffix.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // neg_r_inv as uint256 (BE bytes → buffer)
        let neg_r_inv_buf = self.device.new_buffer_with_data(
            neg_r_inv.as_ptr() as *const _,
            32,
            MTLResourceOptions::StorageModeShared,
        );

        // u2r point (x, y as BE bytes → AffinePoint struct)
        let mut u2r_data = [0u8; 64];
        u2r_data[..32].copy_from_slice(u2r_x);
        u2r_data[32..].copy_from_slice(u2r_y);
        let u2r_buf = self.device.new_buffer_with_data(
            u2r_data.as_ptr() as *const _,
            64,
            MTLResourceOptions::StorageModeShared,
        );

        // Hit output buffers
        let hit_count_buf = self.device.new_buffer(
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            *(hit_count_buf.contents() as *mut u32) = 0;
        }

        let hit_indices_buf = self.device.new_buffer(
            (MAX_HITS * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Dispatch
        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pinning_pipeline);
        encoder.set_buffer(0, Some(&params_buf), 0);
        encoder.set_buffer(1, Some(&suffix_buf), 0);
        encoder.set_buffer(2, Some(&neg_r_inv_buf), 0);
        encoder.set_buffer(3, Some(&u2r_buf), 0);
        encoder.set_buffer(4, Some(&self.gtable_x_buf), 0);
        encoder.set_buffer(5, Some(&self.gtable_y_buf), 0);
        encoder.set_buffer(6, Some(&hit_count_buf), 0);
        encoder.set_buffer(7, Some(&hit_indices_buf), 0);

        let grid = MTLSize::new(batch_size as u64, 1, 1);
        let threadgroup = MTLSize::new(THREADS_PER_GROUP, 1, 1);
        encoder.dispatch_threads(grid, threadgroup);
        encoder.end_encoding();

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read results
        let count = unsafe { *(hit_count_buf.contents() as *const u32) };
        let count = (count as usize).min(MAX_HITS);

        let mut hits = Vec::with_capacity(count);
        let indices_ptr = hit_indices_buf.contents() as *const u32;
        for i in 0..count {
            let thread_idx = unsafe { *indices_ptr.add(i) };
            hits.push(GpuPinningHit {
                locktime: start_lt + thread_idx,
                thread_index: thread_idx,
            });
        }

        hits
    }

    /// Raw dispatch of a pinning-style kernel with explicit pipeline and threadgroup size.
    #[allow(clippy::too_many_arguments)]
    pub fn search_pinning_batch_raw(
        &self,
        pipeline: &ComputePipelineState,
        midstate: &[u32; 8],
        suffix: &[u8],
        total_preimage_len: u32,
        seq_offset: u32,
        lt_offset: u32,
        seq_value: u32,
        start_lt: u32,
        batch_size: u32,
        neg_r_inv: &[u8; 32],
        u2r_x: &[u8; 32],
        u2r_y: &[u8; 32],
        easy_mode: bool,
        threadgroup_size: u64,
    ) -> Vec<GpuPinningHit> {
        let params = PinningParamsGpu {
            midstate: *midstate,
            total_preimage_len,
            suffix_len: suffix.len() as u32,
            seq_offset,
            lt_offset,
            seq_value,
            start_lt,
            easy_mode: if easy_mode { 1 } else { 0 },
            _pad: 0,
        };

        let params_buf = self.device.new_buffer_with_data(
            &params as *const PinningParamsGpu as *const _,
            mem::size_of::<PinningParamsGpu>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let suffix_buf = self.device.new_buffer_with_data(
            suffix.as_ptr() as *const _, suffix.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let neg_r_inv_buf = self.device.new_buffer_with_data(
            neg_r_inv.as_ptr() as *const _, 32, MTLResourceOptions::StorageModeShared,
        );
        let mut u2r_data = [0u8; 64];
        u2r_data[..32].copy_from_slice(u2r_x);
        u2r_data[32..].copy_from_slice(u2r_y);
        let u2r_buf = self.device.new_buffer_with_data(
            u2r_data.as_ptr() as *const _, 64, MTLResourceOptions::StorageModeShared,
        );
        let hit_count_buf = self.device.new_buffer(
            mem::size_of::<u32>() as u64, MTLResourceOptions::StorageModeShared,
        );
        unsafe { *(hit_count_buf.contents() as *mut u32) = 0; }
        let hit_indices_buf = self.device.new_buffer(
            (MAX_HITS * mem::size_of::<u32>()) as u64, MTLResourceOptions::StorageModeShared,
        );

        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&params_buf), 0);
        encoder.set_buffer(1, Some(&suffix_buf), 0);
        encoder.set_buffer(2, Some(&neg_r_inv_buf), 0);
        encoder.set_buffer(3, Some(&u2r_buf), 0);
        encoder.set_buffer(4, Some(&self.gtable_x_buf), 0);
        encoder.set_buffer(5, Some(&self.gtable_y_buf), 0);
        encoder.set_buffer(6, Some(&hit_count_buf), 0);
        encoder.set_buffer(7, Some(&hit_indices_buf), 0);

        let tg = threadgroup_size.min(pipeline.max_total_threads_per_threadgroup());
        encoder.dispatch_threads(
            MTLSize::new(batch_size as u64, 1, 1),
            MTLSize::new(tg, 1, 1),
        );
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let count = unsafe { *(hit_count_buf.contents() as *const u32) };
        let count = (count as usize).min(MAX_HITS);
        let mut hits = Vec::with_capacity(count);
        let indices_ptr = hit_indices_buf.contents() as *const u32;
        for i in 0..count {
            let thread_idx = unsafe { *indices_ptr.add(i) };
            hits.push(GpuPinningHit {
                locktime: start_lt + thread_idx,
                thread_index: thread_idx,
            });
        }
        hits
    }

    /// Run the per-thread batched pinning kernel (Montgomery batch inversion).
    ///
    /// Each thread processes BATCH_N=4 candidates, amortizing the field_inv
    /// across the group. `batch_size` must be a multiple of 4.
    pub fn search_pinning_batched(
        &self,
        midstate: &[u32; 8],
        suffix: &[u8],
        total_preimage_len: u32,
        seq_offset: u32,
        lt_offset: u32,
        seq_value: u32,
        start_lt: u32,
        batch_size: u32,
        neg_r_inv: &[u8; 32],
        u2r_x: &[u8; 32],
        u2r_y: &[u8; 32],
        easy_mode: bool,
    ) -> Vec<GpuPinningHit> {
        const BATCH_N: u32 = 8;
        assert!(batch_size % BATCH_N == 0, "batch_size must be a multiple of {BATCH_N}");

        let pipeline = self.make_pipeline("pinning_search_batched");

        let params = PinningParamsGpu {
            midstate: *midstate,
            total_preimage_len,
            suffix_len: suffix.len() as u32,
            seq_offset,
            lt_offset,
            seq_value,
            start_lt,
            easy_mode: if easy_mode { 1 } else { 0 },
            _pad: 0,
        };

        let params_buf = self.device.new_buffer_with_data(
            &params as *const PinningParamsGpu as *const _,
            mem::size_of::<PinningParamsGpu>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let suffix_buf = self.device.new_buffer_with_data(
            suffix.as_ptr() as *const _, suffix.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let neg_r_inv_buf = self.device.new_buffer_with_data(
            neg_r_inv.as_ptr() as *const _, 32,
            MTLResourceOptions::StorageModeShared,
        );
        let mut u2r_data = [0u8; 64];
        u2r_data[..32].copy_from_slice(u2r_x);
        u2r_data[32..].copy_from_slice(u2r_y);
        let u2r_buf = self.device.new_buffer_with_data(
            u2r_data.as_ptr() as *const _, 64,
            MTLResourceOptions::StorageModeShared,
        );
        let hit_count_buf = self.device.new_buffer(
            mem::size_of::<u32>() as u64, MTLResourceOptions::StorageModeShared,
        );
        unsafe { *(hit_count_buf.contents() as *mut u32) = 0; }
        let hit_indices_buf = self.device.new_buffer(
            (MAX_HITS * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&params_buf), 0);
        encoder.set_buffer(1, Some(&suffix_buf), 0);
        encoder.set_buffer(2, Some(&neg_r_inv_buf), 0);
        encoder.set_buffer(3, Some(&u2r_buf), 0);
        encoder.set_buffer(4, Some(&self.gtable_x_buf), 0);
        encoder.set_buffer(5, Some(&self.gtable_y_buf), 0);
        encoder.set_buffer(6, Some(&hit_count_buf), 0);
        encoder.set_buffer(7, Some(&hit_indices_buf), 0);

        // Each thread processes BATCH_N candidates
        let grid_threads = batch_size / BATCH_N;
        let tg = THREADS_PER_GROUP.min(pipeline.max_total_threads_per_threadgroup());
        encoder.dispatch_threads(
            MTLSize::new(grid_threads as u64, 1, 1),
            MTLSize::new(tg, 1, 1),
        );
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let count = unsafe { *(hit_count_buf.contents() as *const u32) };
        let count = (count as usize).min(MAX_HITS);
        let mut hits = Vec::with_capacity(count);
        let indices_ptr = hit_indices_buf.contents() as *const u32;
        for i in 0..count {
            let candidate_idx = unsafe { *indices_ptr.add(i) };
            hits.push(GpuPinningHit {
                locktime: start_lt + candidate_idx,
                thread_index: candidate_idx,
            });
        }
        hits
    }

    /// Run a digest search batch on the GPU.
    ///
    /// Evaluates `num_candidates` subsets against the hash-to-sig puzzle.
    /// Each candidate's subset indices are taken from `subsets[i*t .. i*t + t]`.
    /// Returns the indices (within the batch) of subsets that produced hits.
    pub fn search_digest_batch(
        &self,
        params: &GpuDigestSearchParams,
        subsets: &[u32],
        t: u32,
        n: u32,
        num_candidates: u32,
        easy_mode: bool,
    ) -> Vec<u32> {
        assert_eq!(
            subsets.len(), (num_candidates * t) as usize,
            "subsets buffer must have num_candidates*t entries"
        );

        #[repr(C)]
        struct DigestParamsGpu {
            midstate: [u32; 8],
            total_preimage_len: u32,
            base_tail_len: u32,
            dummy_push_len: u32,
            t: u32,
            n: u32,
            easy_mode: u32,
            _pad: u32,
        }

        // After FindAndDelete of selected dummies, each subset removes t*dummy_push_len bytes.
        let bytes_removed = t * params.dummy_push_len;
        let total_preimage_len = params.prefix_len + params.base_tail.len() as u32 - bytes_removed;

        let gpu_params = DigestParamsGpu {
            midstate: params.midstate,
            total_preimage_len,
            base_tail_len: params.base_tail.len() as u32,
            dummy_push_len: params.dummy_push_len,
            t,
            n,
            easy_mode: if easy_mode { 1 } else { 0 },
            _pad: 0,
        };

        let pipeline = self.make_pipeline("digest_search");

        let params_buf = self.device.new_buffer_with_data(
            &gpu_params as *const DigestParamsGpu as *const _,
            mem::size_of::<DigestParamsGpu>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let base_tail_buf = self.device.new_buffer_with_data(
            params.base_tail.as_ptr() as *const _,
            params.base_tail.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let dummy_offsets_buf = self.device.new_buffer_with_data(
            params.dummy_offsets.as_ptr() as *const _,
            (params.dummy_offsets.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let subsets_buf = self.device.new_buffer_with_data(
            subsets.as_ptr() as *const _,
            (subsets.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let neg_r_inv_buf = self.device.new_buffer_with_data(
            params.neg_r_inv.as_ptr() as *const _, 32,
            MTLResourceOptions::StorageModeShared,
        );
        let mut u2r_data = [0u8; 64];
        u2r_data[..32].copy_from_slice(&params.u2r_x);
        u2r_data[32..].copy_from_slice(&params.u2r_y);
        let u2r_buf = self.device.new_buffer_with_data(
            u2r_data.as_ptr() as *const _, 64,
            MTLResourceOptions::StorageModeShared,
        );
        let hit_count_buf = self.device.new_buffer(
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe { *(hit_count_buf.contents() as *mut u32) = 0; }
        let hit_indices_buf = self.device.new_buffer(
            (MAX_HITS * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&params_buf), 0);
        encoder.set_buffer(1, Some(&base_tail_buf), 0);
        encoder.set_buffer(2, Some(&dummy_offsets_buf), 0);
        encoder.set_buffer(3, Some(&subsets_buf), 0);
        encoder.set_buffer(4, Some(&neg_r_inv_buf), 0);
        encoder.set_buffer(5, Some(&u2r_buf), 0);
        encoder.set_buffer(6, Some(&self.gtable_x_buf), 0);
        encoder.set_buffer(7, Some(&self.gtable_y_buf), 0);
        encoder.set_buffer(8, Some(&hit_count_buf), 0);
        encoder.set_buffer(9, Some(&hit_indices_buf), 0);

        let tg = THREADS_PER_GROUP.min(pipeline.max_total_threads_per_threadgroup());
        encoder.dispatch_threads(
            MTLSize::new(num_candidates as u64, 1, 1),
            MTLSize::new(tg, 1, 1),
        );
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let count = unsafe { *(hit_count_buf.contents() as *const u32) };
        let count = (count as usize).min(MAX_HITS);
        let mut hits = Vec::with_capacity(count);
        let indices_ptr = hit_indices_buf.contents() as *const u32;
        for i in 0..count {
            hits.push(unsafe { *indices_ptr.add(i) });
        }
        hits
    }

    /// Run digest search with on-GPU nth_combination computation.
    ///
    /// Instead of precomputing subsets on CPU, each GPU thread computes its
    /// own subset via nth_combination(n, t, start_index + gid) using a
    /// binomial coefficient table. Eliminates CPU preprocessing overhead.
    ///
    /// Returns global candidate indices (u64) — the absolute position in
    /// the C(n,t) search space. Callers can use these directly to look up
    /// the corresponding subset via `subset::nth_combination`.
    #[allow(clippy::too_many_arguments)]
    pub fn search_digest_batch_nth(
        &self,
        params: &GpuDigestSearchParams,
        t: u32,
        n: u32,
        start_index: u64,
        num_candidates: u32,
        easy_mode: bool,
    ) -> Vec<u64> {
        #[repr(C)]
        struct DigestNthParamsGpu {
            midstate: [u32; 8],
            total_preimage_len: u32,
            base_tail_len: u32,
            dummy_push_len: u32,
            t: u32,
            n: u32,
            easy_mode: u32,
            binom_stride: u32,
            start_index_lo: u32,
            start_index_hi: u32,
            _pad0: u32,
            _pad1: u32,
        }

        // Build binomial coefficient table: rows 0..n, cols 0..=t
        let binom_stride = (t + 1) as usize;
        let mut binom_table = vec![0u64; (n as usize) * binom_stride];
        for row in 0..(n as usize) {
            for col in 0..=(t as usize) {
                let coef = subset::binomial_coefficient(row, col);
                binom_table[row * binom_stride + col] = coef.min(u64::MAX as u128) as u64;
            }
        }

        let bytes_removed = t * params.dummy_push_len;
        let total_preimage_len = params.prefix_len + params.base_tail.len() as u32 - bytes_removed;

        let gpu_params = DigestNthParamsGpu {
            midstate: params.midstate,
            total_preimage_len,
            base_tail_len: params.base_tail.len() as u32,
            dummy_push_len: params.dummy_push_len,
            t,
            n,
            easy_mode: if easy_mode { 1 } else { 0 },
            binom_stride: binom_stride as u32,
            start_index_lo: (start_index & 0xFFFFFFFF) as u32,
            start_index_hi: (start_index >> 32) as u32,
            _pad0: 0,
            _pad1: 0,
        };

        let pipeline = self.make_pipeline("digest_search_nth");

        let params_buf = self.device.new_buffer_with_data(
            &gpu_params as *const DigestNthParamsGpu as *const _,
            mem::size_of::<DigestNthParamsGpu>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let base_tail_buf = self.device.new_buffer_with_data(
            params.base_tail.as_ptr() as *const _,
            params.base_tail.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let dummy_offsets_buf = self.device.new_buffer_with_data(
            params.dummy_offsets.as_ptr() as *const _,
            (params.dummy_offsets.len() * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let binom_buf = self.device.new_buffer_with_data(
            binom_table.as_ptr() as *const _,
            (binom_table.len() * mem::size_of::<u64>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let neg_r_inv_buf = self.device.new_buffer_with_data(
            params.neg_r_inv.as_ptr() as *const _, 32,
            MTLResourceOptions::StorageModeShared,
        );
        let mut u2r_data = [0u8; 64];
        u2r_data[..32].copy_from_slice(&params.u2r_x);
        u2r_data[32..].copy_from_slice(&params.u2r_y);
        let u2r_buf = self.device.new_buffer_with_data(
            u2r_data.as_ptr() as *const _, 64,
            MTLResourceOptions::StorageModeShared,
        );
        let hit_count_buf = self.device.new_buffer(
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe { *(hit_count_buf.contents() as *mut u32) = 0; }
        let hit_indices_buf = self.device.new_buffer(
            (MAX_HITS * mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&params_buf), 0);
        encoder.set_buffer(1, Some(&base_tail_buf), 0);
        encoder.set_buffer(2, Some(&dummy_offsets_buf), 0);
        encoder.set_buffer(3, Some(&binom_buf), 0);
        encoder.set_buffer(4, Some(&neg_r_inv_buf), 0);
        encoder.set_buffer(5, Some(&u2r_buf), 0);
        encoder.set_buffer(6, Some(&self.gtable_x_buf), 0);
        encoder.set_buffer(7, Some(&self.gtable_y_buf), 0);
        encoder.set_buffer(8, Some(&hit_count_buf), 0);
        encoder.set_buffer(9, Some(&hit_indices_buf), 0);

        let tg = THREADS_PER_GROUP.min(pipeline.max_total_threads_per_threadgroup());
        encoder.dispatch_threads(
            MTLSize::new(num_candidates as u64, 1, 1),
            MTLSize::new(tg, 1, 1),
        );
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let count = unsafe { *(hit_count_buf.contents() as *const u32) };
        let count = (count as usize).min(MAX_HITS);
        let mut hits: Vec<u64> = Vec::with_capacity(count);
        let indices_ptr = hit_indices_buf.contents() as *const u32;
        for i in 0..count {
            let local = unsafe { *indices_ptr.add(i) } as u64;
            hits.push(start_index + local);
        }
        hits
    }

    /// Get the Metal device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the command queue.
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Get the device name.
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Get the maximum threads per threadgroup for the pinning pipeline.
    pub fn max_threads_per_group(&self) -> u64 {
        self.pinning_pipeline.max_total_threads_per_threadgroup()
    }

    /// Create a compute pipeline for a named kernel. Cache externally for repeated use.
    pub fn make_pipeline(&self, kernel_name: &str) -> ComputePipelineState {
        let function = self.library.get_function(kernel_name, None)
            .unwrap_or_else(|_| panic!("kernel not found: {kernel_name}"));
        self.device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap_or_else(|_| panic!("pipeline failed: {kernel_name}"))
    }

    /// Benchmark kernel: run `iterations` across `num_threads` GPU threads.
    pub fn bench_field_op(&self, kernel_name: &str, num_threads: u32, iterations: u32) {
        let pipeline = self.make_pipeline(kernel_name);
        self.run_pipeline(&pipeline, num_threads, iterations);
    }

    /// Run a pre-built pipeline with the standard seed/out/iterations buffer layout.
    pub fn run_pipeline(&self, pipeline: &ComputePipelineState, num_threads: u32, iterations: u32) {
        let seed: [u8; 32] = [0x42; 32];
        let seed_buf = self.device.new_buffer_with_data(
            seed.as_ptr() as *const _, 32, MTLResourceOptions::StorageModeShared,
        );
        let out_buf = self.device.new_buffer(32, MTLResourceOptions::StorageModeShared);
        let iter_buf = self.device.new_buffer_with_data(
            &iterations as *const u32 as *const _, 4, MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&seed_buf), 0);
        enc.set_buffer(1, Some(&out_buf), 0);
        enc.set_buffer(2, Some(&iter_buf), 0);
        let tg = THREADS_PER_GROUP.min(pipeline.max_total_threads_per_threadgroup());
        enc.dispatch_threads(
            MTLSize::new(num_threads as u64, 1, 1),
            MTLSize::new(tg, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // ============================================================
    // Test helpers — dispatch individual GPU operations for correctness verification
    // ============================================================

    /// Run SHA-256 on 32 bytes using the GPU and return the 32-byte hash.
    pub fn test_sha256(&self, input: &[u8; 32]) -> [u8; 32] {
        self.run_simple_kernel("test_sha256", input, 32)
            .try_into()
            .expect("32 bytes")
    }

    /// Run HASH160 on 33 bytes using the GPU and return the 20-byte hash.
    pub fn test_hash160(&self, input: &[u8; 33]) -> [u8; 20] {
        self.run_simple_kernel("test_hash160", input, 20)
            .try_into()
            .expect("20 bytes")
    }

    /// Run field multiplication (a * b mod P) on the GPU.
    /// All values are 32 big-endian bytes.
    pub fn test_field_mul(&self, a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
        let function = self.library.get_function("test_field_mul", None).expect("fn");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .expect("pipeline");

        let a_buf = self.device.new_buffer_with_data(
            a.as_ptr() as *const _, 32, MTLResourceOptions::StorageModeShared,
        );
        let b_buf = self.device.new_buffer_with_data(
            b.as_ptr() as *const _, 32, MTLResourceOptions::StorageModeShared,
        );
        let out_buf = self.device.new_buffer(32, MTLResourceOptions::StorageModeShared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&a_buf), 0);
        enc.set_buffer(1, Some(&b_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut result = [0u8; 32];
        unsafe {
            std::ptr::copy_nonoverlapping(out_buf.contents() as *const u8, result.as_mut_ptr(), 32);
        }
        result
    }

    /// Run field inversion on the GPU. Returns (inv, a * inv) — second should be [0..0, 1].
    pub fn test_field_inv(&self, a: &[u8; 32]) -> ([u8; 32], [u8; 32]) {
        let function = self.library.get_function("test_field_inv", None).expect("fn");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .expect("pipeline");

        let a_buf = self.device.new_buffer_with_data(
            a.as_ptr() as *const _, 32, MTLResourceOptions::StorageModeShared,
        );
        let inv_buf = self.device.new_buffer(32, MTLResourceOptions::StorageModeShared);
        let prod_buf = self.device.new_buffer(32, MTLResourceOptions::StorageModeShared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&a_buf), 0);
        enc.set_buffer(1, Some(&inv_buf), 0);
        enc.set_buffer(2, Some(&prod_buf), 0);
        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut inv = [0u8; 32];
        let mut prod = [0u8; 32];
        unsafe {
            std::ptr::copy_nonoverlapping(inv_buf.contents() as *const u8, inv.as_mut_ptr(), 32);
            std::ptr::copy_nonoverlapping(prod_buf.contents() as *const u8, prod.as_mut_ptr(), 32);
        }
        (inv, prod)
    }

    /// Run EC scalar multiplication: scalar * G via GTable.
    /// Returns uncompressed point (x, y) as 32 BE bytes each.
    pub fn test_ec_mul(&self, scalar: &[u8; 32]) -> ([u8; 32], [u8; 32]) {
        let function = self.library.get_function("test_ec_mul", None).expect("fn");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .expect("pipeline");

        let scalar_buf = self.device.new_buffer_with_data(
            scalar.as_ptr() as *const _, 32, MTLResourceOptions::StorageModeShared,
        );
        let out_x = self.device.new_buffer(32, MTLResourceOptions::StorageModeShared);
        let out_y = self.device.new_buffer(32, MTLResourceOptions::StorageModeShared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&scalar_buf), 0);
        enc.set_buffer(1, Some(&self.gtable_x_buf), 0);
        enc.set_buffer(2, Some(&self.gtable_y_buf), 0);
        enc.set_buffer(3, Some(&out_x), 0);
        enc.set_buffer(4, Some(&out_y), 0);
        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut x = [0u8; 32];
        let mut y = [0u8; 32];
        unsafe {
            std::ptr::copy_nonoverlapping(out_x.contents() as *const u8, x.as_mut_ptr(), 32);
            std::ptr::copy_nonoverlapping(out_y.contents() as *const u8, y.as_mut_ptr(), 32);
        }
        (x, y)
    }

    /// Run the full EC recovery pipeline on the GPU.
    /// Returns (compressed_pubkey[33], hash160[20]).
    pub fn test_ec_recovery(
        &self,
        digest: &[u8; 32],
        neg_r_inv: &[u8; 32],
        u2r_x: &[u8; 32],
        u2r_y: &[u8; 32],
    ) -> ([u8; 33], [u8; 20]) {
        let function = self.library.get_function("test_ec_recovery", None).expect("fn");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .expect("pipeline");

        let digest_buf = self.device.new_buffer_with_data(
            digest.as_ptr() as *const _, 32, MTLResourceOptions::StorageModeShared,
        );
        let nri_buf = self.device.new_buffer_with_data(
            neg_r_inv.as_ptr() as *const _, 32, MTLResourceOptions::StorageModeShared,
        );
        let mut u2r_data = [0u8; 64];
        u2r_data[..32].copy_from_slice(u2r_x);
        u2r_data[32..].copy_from_slice(u2r_y);
        let u2r_buf = self.device.new_buffer_with_data(
            u2r_data.as_ptr() as *const _, 64, MTLResourceOptions::StorageModeShared,
        );
        let pubkey_buf = self.device.new_buffer(33, MTLResourceOptions::StorageModeShared);
        let h160_buf = self.device.new_buffer(20, MTLResourceOptions::StorageModeShared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&digest_buf), 0);
        enc.set_buffer(1, Some(&nri_buf), 0);
        enc.set_buffer(2, Some(&u2r_buf), 0);
        enc.set_buffer(3, Some(&self.gtable_x_buf), 0);
        enc.set_buffer(4, Some(&self.gtable_y_buf), 0);
        enc.set_buffer(5, Some(&pubkey_buf), 0);
        enc.set_buffer(6, Some(&h160_buf), 0);
        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut pubkey = [0u8; 33];
        let mut h160 = [0u8; 20];
        unsafe {
            std::ptr::copy_nonoverlapping(pubkey_buf.contents() as *const u8, pubkey.as_mut_ptr(), 33);
            std::ptr::copy_nonoverlapping(h160_buf.contents() as *const u8, h160.as_mut_ptr(), 20);
        }
        (pubkey, h160)
    }

    /// Helper: run a kernel with single input/output byte buffers.
    fn run_simple_kernel(&self, name: &str, input: &[u8], output_len: usize) -> Vec<u8> {
        let function = self.library.get_function(name, None).expect("kernel fn");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .expect("pipeline");

        let in_buf = self.device.new_buffer_with_data(
            input.as_ptr() as *const _,
            input.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = self.device.new_buffer(
            output_len as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&in_buf), 0);
        enc.set_buffer(1, Some(&out_buf), 0);
        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut result = vec![0u8; output_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                out_buf.contents() as *const u8,
                result.as_mut_ptr(),
                output_len,
            );
        }
        result
    }

    /// Helper: run a kernel with two input byte buffers and one output byte buffer.
    pub fn run_simple_kernel_2in_1out(
        &self, name: &str, input_a: &[u8], input_b: &[u8], output_len: usize,
    ) -> Vec<u8> {
        let function = self.library.get_function(name, None).expect("kernel fn");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .expect("pipeline");

        let a_buf = self.device.new_buffer_with_data(
            input_a.as_ptr() as *const _, input_a.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let b_buf = self.device.new_buffer_with_data(
            input_b.as_ptr() as *const _, input_b.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = self.device.new_buffer(
            output_len as u64, MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&a_buf), 0);
        enc.set_buffer(1, Some(&b_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut result = vec![0u8; output_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                out_buf.contents() as *const u8, result.as_mut_ptr(), output_len,
            );
        }
        result
    }
}
