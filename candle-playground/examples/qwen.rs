// See: https://github.com/huggingface/candle/blob/0.7.1/candle-examples/examples/qwen/main.rs

extern crate candle_playground;

use anyhow::{Error, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::ModelForCausalLM;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use candle_playground::pipelines::lm::{Args, TextGeneration};

fn main() -> Result<()> {
    let args = Args::parse();
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        format!("Qwen/Qwen2.5-1.5B-Instruct"),
        RepoType::Model,
        "main".to_string(),
    ));
    let tokenizer = Tokenizer::from_file(repo.get("tokenizer.json")?).map_err(Error::msg)?;
    let device = Device::Cpu;
    let model = ModelForCausalLM::new(
        &serde_json::from_slice(&std::fs::read(repo.get("config.json")?)?)?,
        unsafe {
            VarBuilder::from_mmaped_safetensors(
                &vec![repo.get("model.safetensors")?],
                DType::F32,
                &device,
            )?
        },
    )?;
    let prompt = format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        "You are a helpful assistant.", &args.prompt,
    );
    let mut pipeline = TextGeneration::new(
        &device,
        tokenizer,
        "<|endoftext|>",
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
    )?;
    pipeline.run(model, &prompt, args.sample_len)?;
    Ok(())
}
