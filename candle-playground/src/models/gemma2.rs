use std::fs::File;

use anyhow::Result;
use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::models::gemma2::Model;

pub enum Gemma {
    Full(Model),
    Quantized(ModelWeights),
}

pub struct ModelWeights {}

impl ModelWeights {
    pub fn from_gguf(
        content: gguf_file::Content,
        file: &mut File,
        device: &Device,
    ) -> Result<Self> {
        println!("{:?}", content.metadata.keys());
        Ok(Self {})
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        todo!()
    }
}
