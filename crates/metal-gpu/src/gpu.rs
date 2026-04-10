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
}
