// See: https://github.com/huggingface/candle/blob/0.7.1/candle-examples/examples/quantized/main.rs

extern crate candle_playground;

use anyhow::Error;
use candle_core::{quantized::gguf_file, Device};
use candle_transformers::models::quantized_llama::ModelWeights;
use clap::Parser;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use candle_playground::pipelines::lm::{Args, TextGeneration};

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let api = Api::new()?;
    let tokenizer = Tokenizer::from_file(
        api.model("rinna/youri-7b".to_string())
            .get("tokenizer.json")?,
    )
    .map_err(Error::msg)?;
    let device = Device::Cpu;
    let model_path = api
        .repo(hf_hub::Repo::with_revision(
            "RichardErkhov/moneyforward_-_houou-instruction-7b-v3-gguf".to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ))
        .get("houou-instruction-7b-v3.Q4_K_M.gguf")?;
    let mut model_file = std::fs::File::open(&model_path)?;
    let model_content =
        gguf_file::Content::read(&mut model_file).map_err(|e| e.with_path(model_path))?;
    let model = ModelWeights::from_gguf(model_content, &mut model_file, &device)?;
    let prompt = format!("以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{}\n\n### 応答:\n", args.prompt);
    let mut pipeline = TextGeneration::new(
        &device,
        tokenizer,
        "</s>",
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
    )?;
    pipeline.run(model, &prompt, args.sample_len)?;
    Ok(())
}
