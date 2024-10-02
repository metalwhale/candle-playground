// See: https://github.com/huggingface/candle/blob/0.7.1/candle-examples/examples/quantized/main.rs

extern crate candle_playground;

use std::io::Write;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_llama as model;
use clap::Parser;
use model::ModelWeights;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The initial prompt, use 'interactive' for entering multiple prompts in an interactive way
    /// and 'chat' for an interactive model where history of previous prompts and generated tokens
    /// is preserved.
    #[arg(long)]
    prompt: String,

    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 1000)]
    sample_len: usize,

    /// The tokenizer config in json format.
    #[arg(long)]
    tokenizer: Option<String>,

    /// The temperature used to generate samples, use 0 for greedy sampling.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

impl Args {
    fn tokenizer(&self) -> anyhow::Result<Tokenizer> {
        let tokenizer_path = match &self.tokenizer {
            Some(config) => std::path::PathBuf::from(config),
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let repo = "rinna/youri-7b";
                let api = api.model(repo.to_string());
                api.get("tokenizer.json")?
            }
        };
        Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)
    }

    fn model(&self) -> anyhow::Result<std::path::PathBuf> {
        let api = hf_hub::api::sync::Api::new()?;
        let model_path = api
            .repo(hf_hub::Repo::with_revision(
                "RichardErkhov/moneyforward_-_houou-instruction-7b-v3-gguf".to_string(),
                hf_hub::RepoType::Model,
                "main".to_string(),
            ))
            .get("houou-instruction-7b-v3.Q4_K_M.gguf")?;
        Ok(model_path)
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let model_path = args.model()?;
    let mut file = std::fs::File::open(&model_path)?;
    let model_content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
    let device = Device::Cpu;
    let mut model = ModelWeights::from_gguf(model_content, &mut file, &device)?;

    let tokenizer = args.tokenizer()?;
    let mut tos = TokenOutputStream::new(tokenizer);

    let pre_prompt_tokens = vec![];
    let prompt_str = format!("以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{}\n\n### 応答:\n", args.prompt);
    print!("{}", &prompt_str);
    let tokens = tos
        .tokenizer()
        .encode(prompt_str, true)
        .map_err(anyhow::Error::msg)?;

    let prompt_tokens = [&pre_prompt_tokens, tokens.get_ids()].concat();
    let to_sample = args.sample_len.saturating_sub(1);
    let prompt_tokens = if prompt_tokens.len() + to_sample > model::MAX_SEQ_LEN - 10 {
        let to_remove = prompt_tokens.len() + to_sample + 10 - model::MAX_SEQ_LEN;
        prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
    } else {
        prompt_tokens
    };
    let mut all_tokens = vec![];
    let mut logits_processor = {
        let temperature = args.temperature;
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (args.top_k, args.top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        LogitsProcessor::from_sampling(rand::random(), sampling)
    };

    let mut next_token = {
        let input = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?;
        logits_processor.sample(&logits)?
    };
    all_tokens.push(next_token);
    if let Some(t) = tos.next_token(next_token)? {
        print!("{t}");
        std::io::stdout().flush()?;
    }

    let eos_token = "</s>";

    let eos_token = *tos.tokenizer().get_vocab(true).get(eos_token).unwrap();
    for index in 0..to_sample {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt_tokens.len() + index)?;
        let logits = logits.squeeze(0)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = all_tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &all_tokens[start_at..],
            )?
        };
        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
        if next_token == eos_token {
            break;
        };
    }
    if let Some(rest) = tos.decode_rest().map_err(candle_core::Error::msg)? {
        print!("{rest}");
    }
    std::io::stdout().flush()?;

    Ok(())
}
