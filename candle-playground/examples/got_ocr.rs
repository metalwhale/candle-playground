extern crate candle_playground;

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};

use candle_playground::pipelines::ocr::Args;

fn main() -> Result<()> {
    let args = Args::parse();
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        format!("stepfun-ai/GOT-OCR2_0"),
        RepoType::Model,
        "main".to_string(),
    ));
    unsafe {
        VarBuilder::from_mmaped_safetensors(
            &vec![repo.get("model.safetensors")?],
            DType::F32,
            &Device::Cpu,
        )?
        .contains_tensor("model.embed_tokens.weight")
    };
    println!("{:?}", args.image_path);
    Ok(())
}
