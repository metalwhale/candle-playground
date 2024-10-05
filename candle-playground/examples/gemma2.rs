use std::{env, fs::File};

use anyhow::{Error, Result};
use candle_core::{quantized::gguf_file, DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma2::Model;
use clap::Parser;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;

use candle_playground::{
    models::gemma2::{Gemma, ModelWeights},
    pipelines::lm::{Args, TextGeneration},
};

fn main() -> Result<()> {
    let args = Args::parse();
    let api = ApiBuilder::new()
        .with_token(env::var("HF_TOKEN").ok())
        .build()?;
    let repo = api.repo(Repo::with_revision(
        "google/gemma-2-2b-jpn-it".to_string(),
        RepoType::Model,
        "main".to_string(),
    ));
    let tokenizer = Tokenizer::from_file(repo.get("tokenizer.json")?).map_err(Error::msg)?;
    let device = Device::Cpu;
    let model: Gemma = if args.use_quantized {
        let model_path = api
            .repo(hf_hub::Repo::with_revision(
                "QuantFactory/gemma-2-2b-jpn-it-GGUF".to_string(),
                hf_hub::RepoType::Model,
                "main".to_string(),
            ))
            .get("gemma-2-2b-jpn-it.Q4_K_M.gguf")?;
        let mut model_file = File::open(&model_path)?;
        let model_content =
            gguf_file::Content::read(&mut model_file).map_err(|e| e.with_path(model_path))?;
        Gemma::Quantized(ModelWeights::from_gguf(
            model_content,
            &mut model_file,
            &device,
        )?)
    } else {
        Gemma::Full(Model::new(
            false,
            &serde_json::from_reader(std::fs::File::open(repo.get("config.json")?)?)?,
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?,
                    DType::F32,
                    &device,
                )?
            },
        )?)
    };
    let prompt = format!(
        "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
        args.prompt,
    );
    let pipeline = TextGeneration::new(
        &device,
        tokenizer,
        "<eos>",
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
    );
    pipeline?.run(model, &prompt, args.sample_len)?;
    Ok(())
}
