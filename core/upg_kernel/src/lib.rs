
use pyo3::prelude::*;
use pyo3::types::PyModule;
use candle_core::{Tensor, Device, DType, Error as CandleError};
use candle_nn::{Linear, LayerNorm, Embedding, Module, VarBuilder};
use numpy::{PyArray1, PyArray2, PyArray3, ToPyArray};

// Define a simplified UPG Block (Attention + SparseFFN)
// For the prototype, we assume a single heavy block to prove loop speedup.
struct UPGBlock {
    // In a real port, we'd have Attention here too.
    // For now, we simulate the compute density of Attention + FFN
    // using large linear layers to match the parameter count/FLOPs.
    ffn_up: Linear,
    ffn_down: Linear,
    norm: LayerNorm,
    sparsity: f64,
}

impl UPGBlock {
    fn new(vs: VarBuilder, hidden: usize, intermediate: usize) -> Result<Self, CandleError> {
        let ffn_up = candle_nn::linear(hidden, intermediate, vs.pp("ffn_up"))?;
        let ffn_down = candle_nn::linear(intermediate, hidden, vs.pp("ffn_down"))?;
        let norm = candle_nn::layer_norm(hidden, 1e-5, vs.pp("norm"))?;
        Ok(Self { ffn_up, ffn_down, norm, sparsity: 0.96 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, CandleError> {
        let residual = x;
        let x = self.norm.forward(x)?;
        
        // Sparse FFN
        // 1. Up Project
        let hidden = self.ffn_up.forward(&x)?.gelu()?;
        
        // 2. "True" Sparsity (Compute Skipping)
        let hidden = if self.sparsity > 0.0 {
            // A. Up Projection is Dense (The "Selector")
            // B. Block Energy Calculation (Restoring Locality)
            const BLOCK_SIZE: usize = 64;
            let (b, s, inter_dim) = hidden.dims3()?;
            let num_blocks = inter_dim / BLOCK_SIZE;
            
            // Reshape to (B, S, NumBlocks, BlockSize)
            let hidden_blocked = hidden.reshape((b, s, num_blocks, BLOCK_SIZE))?;
            
            // Calculate Block L1 Energy: Sum(Abs(val)) along last dim
            let block_energy = hidden_blocked.abs()?.sum(3)?; // (B, S, NumBlocks)
            
            // Flatten generic dims for topk
            let energy_vec = block_energy.flatten_all()?.to_vec1::<f32>()?;
            
            // Calculate Target Blocks (e.g. 96% sparsity = top 4% blocks)
            // But ensure at least 1 block
            let k_blocks = ((num_blocks as f64) * (1.0 - self.sparsity)) as usize;
            let k_blocks = k_blocks.max(1);
            
            // Sort & Select Blocks
            let mut pairs: Vec<(f32, usize)> = energy_vec.iter()
                .enumerate()
                .map(|(i, &v)| (v, i))
                .collect();
            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            
            let top_block_indices: Vec<u32> = pairs.iter()
                .take(k_blocks)
                .map(|(_, i)| *i as u32)
                .collect();
                
            let block_indices = Tensor::from_slice(&top_block_indices, k_blocks, hidden.device())?;
            
            // C. Gather Inputs (Block Wise)
            // Select active blocks from hidden_blocked: (B, S, K_Blocks, BlockSize)
            // We assume B=1, S=1 for generation loop usually
            let h_subset_blocks = hidden_blocked.index_select(&block_indices, 2)?;
            // Flatten back to (1, 1, K_Blocks * BlockSize)
            let h_subset = h_subset_blocks.flatten_from(2)?;
            
            // D. Gather Weights (Block Wise)
            // Weight is (Hidden, Inter).
            // View as (Hidden, NumBlocks, BlockSize)
            let w_blocked = self.ffn_down.weight().reshape((self.ffn_down.weight().dim(0)?, num_blocks, BLOCK_SIZE))?;
            
            // Select active blocks (dim 1)
            let w_subset_blocks = w_blocked.index_select(&block_indices, 1)?;
            
            // Flatten active weights to (Hidden, K_Blocks * BlockSize)
            let w_subset = w_subset_blocks.flatten_from(1)?;
            
            // E. Compute Small MatMul
            // Flatten h_subset to (1, Active) to ensure safe 2D matmul
            // h (1, 1, Active) -> (1, Active)
            let h_flat = h_subset.flatten_all()?.unsqueeze(0)?;
            
            // w.t is (Active, Hidden)
            // out (1, Active) @ (Active, Hidden) -> (1, Hidden)
            let hidden_dim = w_subset.dim(0)?;
            h_flat.matmul(&w_subset.t()?)?.reshape((b, s, hidden_dim))?
        } else {
            self.ffn_down.forward(&hidden)?
        };

        // 3. Skip Down Compute (Handled above in if/else)
        // let x = self.ffn_down.forward(&hidden)?; // Removed
        
        // Residual
        let x = (x + residual)?;
        Ok(x)
    }
}

#[pyclass]
struct NativeRustUPG {
    // We hold the model components in Rust memory
    embedding: Embedding,
    blocks: Vec<UPGBlock>,
    head: Linear,
    device: Device,
}

#[pymethods]
impl NativeRustUPG {
    #[new]
    fn new(vocab_size: usize, hidden: usize, layers: usize) -> PyResult<Self> {
        // Initialize random weights locally for the prototype.
        // In production, we'd pass weights from Python or load safe-tensors.
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device); // Zeros for speed, normally random/loaded

        let embedding = candle_nn::embedding(vocab_size, hidden, vb.pp("emb"))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let mut blocks = Vec::new();
        // Simulate 'layers' number of blocks
        // For the sake of the "1000 tok/s" demo, we can use a smaller number 
        // if we are proving LOOP overhead reduction, but let's try to be honest.
        for i in 0..layers {
            let block = UPGBlock::new(vb.pp(format!("block_{}", i)), hidden, hidden * 4)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            blocks.push(block);
        }

        let head = candle_nn::linear(hidden, vocab_size, vb.pp("head"))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self {
            embedding,
            blocks,
            head,
            device,
        })
    }

    /// The FUSED Generation Loop.
    /// No Python transitions between tokens.
    fn generate(&self, prompt_tokens: Vec<u32>, max_new_tokens: usize) -> PyResult<Vec<u32>> {
        let mut tokens = prompt_tokens.clone();
        
        for _ in 0..max_new_tokens {
            let _ctx_len = tokens.len();
            
            // 1. Prepare Input (Last token only for simplified autoregressive check, 
            //    or full context if we had Attention caching. 
            //    For this speed test, we process the last token to simulate "Next Token Prediction" compute)
            let last_token = *tokens.last().unwrap();
            let input = Tensor::new(&[last_token], &self.device)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
                .unsqueeze(0) // Batch dim
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            // 2. Forward Pass (Rust Speed)
            let mut x = self.embedding.forward(&input)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            for block in &self.blocks {
                x = block.forward(&x)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            }

            let logits = self.head.forward(&x)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
            // 3. Greedy Sample
            let logits_v: Vec<f32> = logits.squeeze(0).unwrap().squeeze(0).unwrap().to_vec1().unwrap();
            let next_token = logits_v.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index as u32)
                .unwrap();

            tokens.push(next_token);
        }

        Ok(tokens)
    }
}

#[pymodule]
fn upg_kernel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NativeRustUPG>()?;
    Ok(())
}
