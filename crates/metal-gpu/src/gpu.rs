use metal::*;
use std::mem;
use std::path::Path;

const SHADER_SOURCE: &str = include_str!("../shaders/binohash.metal");
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

    /// Get the device name.
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Get the maximum threads per threadgroup for the pinning pipeline.
    pub fn max_threads_per_group(&self) -> u64 {
        self.pinning_pipeline.max_total_threads_per_threadgroup()
    }

    /// Benchmark kernel: run `iterations` of field_mul across `num_threads` GPU threads.
    /// Returns wall-clock time. The result is chain-dependent to prevent dead-code elimination.
    pub fn bench_field_op(&self, kernel_name: &str, num_threads: u32, iterations: u32) {
        let library = self
            .device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .expect("compile");
        let function = library.get_function(kernel_name, None).expect("kernel fn");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .expect("pipeline");

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
        let library = self
            .device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .expect("compile");
        let function = library.get_function("test_field_mul", None).expect("fn");
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
        let library = self
            .device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .expect("compile");
        let function = library.get_function("test_field_inv", None).expect("fn");
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
        let library = self
            .device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .expect("compile");
        let function = library.get_function("test_ec_mul", None).expect("fn");
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
        let library = self
            .device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .expect("compile");
        let function = library.get_function("test_ec_recovery", None).expect("fn");
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
        let library = self
            .device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .expect("compile");
        let function = library.get_function(name, None).expect("kernel fn");
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
}
