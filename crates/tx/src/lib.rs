pub const SIGHASH_ALL: u8 = 0x01;
pub const SIGHASH_NONE: u8 = 0x02;
pub const SIGHASH_SINGLE: u8 = 0x03;
pub const SIGHASH_ANYONECANPAY: u8 = 0x80;

#[derive(Debug, Clone)]
pub struct TxIn {
    pub txid: [u8; 32],
    pub vout: u32,
    pub script_sig: Vec<u8>,
    pub sequence: u32,
}

#[derive(Debug, Clone)]
pub struct TxOut {
    pub value: u64,
    pub script_pubkey: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct Transaction {
    pub version: u32,
    pub inputs: Vec<TxIn>,
    pub outputs: Vec<TxOut>,
    pub locktime: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum TxError {
    #[error("input index {index} out of bounds (tx has {count} inputs)")]
    InputOutOfBounds { index: usize, count: usize },
}

impl Transaction {
    pub fn new(version: u32, locktime: u32) -> Self {
        Self {
            version,
            inputs: Vec::new(),
            outputs: Vec::new(),
            locktime,
        }
    }

    pub fn add_input(&mut self, input: TxIn) {
        self.inputs.push(input);
    }

    pub fn add_output(&mut self, output: TxOut) {
        self.outputs.push(output);
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&serialize_varint(self.inputs.len() as u64));
        for input in &self.inputs {
            out.extend_from_slice(&input.serialize());
        }
        out.extend_from_slice(&serialize_varint(self.outputs.len() as u64));
        for output in &self.outputs {
            out.extend_from_slice(&output.serialize());
        }
        out.extend_from_slice(&self.locktime.to_le_bytes());
        out
    }

    /// Compute the legacy (pre-SegWit) sighash for a specific input.
    ///
    /// `script_code` should already have FindAndDelete applied if needed.
    /// Returns the double-SHA256 of the signing serialization.
    pub fn legacy_sighash(
        &self,
        input_index: usize,
        script_code: &[u8],
        sighash_type: u8,
    ) -> Result<[u8; 32], TxError> {
        let base_type = sighash_type & 0x1f;

        // SIGHASH_SINGLE bug: if input_index >= outputs, return hash of 0x01
        if base_type == SIGHASH_SINGLE && input_index >= self.outputs.len() {
            let mut bug = [0u8; 32];
            bug[31] = 1;
            return Ok(bug);
        }

        let preimage = self.legacy_sighash_preimage(input_index, script_code, sighash_type)?;
        Ok(hash::sha256d(&preimage))
    }

    /// Compute the raw sighash preimage (before double-SHA256).
    ///
    /// Returns the serialized transaction copy with the sighash type appended.
    /// This is the data that gets double-SHA256'd to produce the sighash.
    ///
    /// Does NOT handle the SIGHASH_SINGLE bug case — returns an error if
    /// `input_index >= outputs.len()` with SIGHASH_SINGLE.
    pub fn legacy_sighash_preimage(
        &self,
        input_index: usize,
        script_code: &[u8],
        sighash_type: u8,
    ) -> Result<Vec<u8>, TxError> {
        if input_index >= self.inputs.len() {
            return Err(TxError::InputOutOfBounds {
                index: input_index,
                count: self.inputs.len(),
            });
        }

        let base_type = sighash_type & 0x1f;

        let mut tx_copy = Transaction::new(self.version, self.locktime);

        for (idx, input) in self.inputs.iter().enumerate() {
            if sighash_type & SIGHASH_ANYONECANPAY != 0 && idx != input_index {
                continue;
            }

            let mut copied = TxIn {
                txid: input.txid,
                vout: input.vout,
                script_sig: if idx == input_index {
                    script_code.to_vec()
                } else {
                    Vec::new()
                },
                sequence: input.sequence,
            };

            if matches!(base_type, SIGHASH_NONE | SIGHASH_SINGLE) && idx != input_index {
                copied.sequence = 0;
            }

            tx_copy.add_input(copied);
        }

        match base_type {
            SIGHASH_NONE => {}
            SIGHASH_SINGLE => {
                for output_index in 0..=input_index {
                    if output_index < input_index {
                        tx_copy.add_output(TxOut {
                            value: u64::MAX,
                            script_pubkey: Vec::new(),
                        });
                    } else {
                        tx_copy.add_output(self.outputs[output_index].clone());
                    }
                }
            }
            _ => {
                for output in &self.outputs {
                    tx_copy.add_output(output.clone());
                }
            }
        }

        let mut serialized = tx_copy.serialize();
        serialized.extend_from_slice(&(sighash_type as u32).to_le_bytes());
        Ok(serialized)
    }
}

impl TxIn {
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.txid);
        out.extend_from_slice(&self.vout.to_le_bytes());
        out.extend_from_slice(&serialize_varint(self.script_sig.len() as u64));
        out.extend_from_slice(&self.script_sig);
        out.extend_from_slice(&self.sequence.to_le_bytes());
        out
    }
}

impl TxOut {
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.value as i64).to_le_bytes());
        out.extend_from_slice(&serialize_varint(self.script_pubkey.len() as u64));
        out.extend_from_slice(&self.script_pubkey);
        out
    }
}

pub fn serialize_varint(value: u64) -> Vec<u8> {
    match value {
        0..=0xfc => vec![value as u8],
        0xfd..=0xffff => {
            let mut out = vec![0xfd];
            out.extend_from_slice(&(value as u16).to_le_bytes());
            out
        }
        0x10000..=0xffff_ffff => {
            let mut out = vec![0xfe];
            out.extend_from_slice(&(value as u32).to_le_bytes());
            out
        }
        _ => {
            let mut out = vec![0xff];
            out.extend_from_slice(&value.to_le_bytes());
            out
        }
    }
}
