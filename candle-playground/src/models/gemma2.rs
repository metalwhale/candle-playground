use std::fs::File;

use anyhow::{bail, Result};
use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_nn::{Embedding, Module};
use candle_transformers::models::gemma2::Model;

pub enum Gemma {
    Full(Model),
    Quantized(ModelWeights),
}

// See: https://github.com/huggingface/candle/blob/0.7.1/candle-transformers/src/models/quantized_llama.rs

pub struct ModelWeights {
    token_embd: Embedding,
}

impl ModelWeights {
    pub fn from_gguf(
        content: gguf_file::Content,
        file: &mut File,
        device: &Device,
    ) -> Result<Self> {
        // Read metadata
        let metadata_get = |k: &str| match content.metadata.get(k) {
            None => bail!("Cannot find {k} in metadata"),
            Some(v) => Ok(v),
        };
        let embedding_length = metadata_get("gemma2.embedding_length")?.to_u32()? as usize;
        // Read tensors
        let token_embd = content
            .tensor(file, "token_embd.weight", device)?
            .dequantize(device)?;
        Ok(Self {
            token_embd: Embedding::new(token_embd, embedding_length),
        })
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_, seq_len) = x.dims2()?;
        // TODO: Store masks for reuse
        let mask = Tensor::from_slice(
            &(0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
                .collect::<Vec<u8>>(),
            (seq_len, seq_len),
            x.device(),
        )?;
        let mut layer_input = self.token_embd.forward(x)?;
        println!("{:?}", layer_input.to_vec3::<f32>()?[0][0]);
        todo!()
    }
}
