// See: https://github.com/huggingface/candle/blob/0.7.1/candle-examples/examples/qwen/main.rs

use std::io::Write;

use anyhow::{Error, Result};
use candle_core::{Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::LogitsProcessor;
use clap::Parser;
use tokenizers::Tokenizer;

use crate::models::gemma2::Gemma;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(long)]
    pub prompt: String,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    pub sample_len: usize,

    /// The temperature used to generate samples.
    #[arg(long)]
    pub temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    pub top_p: Option<f64>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    pub repeat_last_n: usize,

    #[arg(long)]
    pub use_quantized: bool,
}

pub trait CausalForward {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor>;
}

pub struct TextGeneration {
    device: Device,
    stream: TokenOutputStream,
    eos_token: u32,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    pub fn new(
        device: &Device,
        tokenizer: Tokenizer,
        eos: &str,
        temperature: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Result<Self> {
        let stream = TokenOutputStream::new(tokenizer);
        let eos_token = match stream.get_token(eos) {
            Some(token) => token,
            None => anyhow::bail!("cannot find the {} token", eos),
        };
        Ok(Self {
            device: device.clone(),
            stream,
            eos_token,
            logits_processor: LogitsProcessor::new(rand::random(), temperature, top_p),
            repeat_penalty,
            repeat_last_n,
        })
    }

    pub fn run<M: CausalForward>(
        &mut self,
        mut model: M,
        prompt: &str,
        sample_len: usize,
    ) -> Result<()> {
        self.stream.clear();
        let mut tokens = self
            .stream
            .tokenizer()
            .encode(prompt, true)
            .map_err(Error::msg)?
            .get_ids()
            .to_vec();
        for &token in tokens.iter() {
            if let Some(text) = self.stream.next_token(token)? {
                print!("{text}")
            }
        }
        let prompt_tokens_len = tokens.len();
        std::io::stdout().flush()?;
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let input = Tensor::new(&tokens[start_pos..], &self.device)?.unsqueeze(0)?;
            let logits = model.forward(&input, start_pos)?;
            let logits = if self.repeat_penalty == 1.0 {
                logits
            } else {
                let start_at = index.saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    // NOTE: Repetitiveness is assessed based only on the generated tokens, not those in the prompt
                    &tokens[prompt_tokens_len + start_at..],
                )?
            };
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            if next_token == self.eos_token {
                break;
            }
            if let Some(text) = self.stream.next_token(next_token)? {
                print!("{text}");
                std::io::stdout().flush()?;
            }
        }
        if let Some(rest) = self.stream.decode_rest().map_err(Error::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        Ok(())
    }
}

impl CausalForward for candle_transformers::models::quantized_llama::ModelWeights {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        Ok(self.forward(input_ids, pos)?.squeeze(0)?)
    }
}

impl CausalForward for candle_transformers::models::qwen2::ModelForCausalLM {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        Ok(self.forward(input_ids, pos)?.squeeze(0)?.squeeze(0)?)
    }
}

impl CausalForward for Gemma {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        Ok(match self {
            Gemma::Full(model) => model.forward(input_ids, pos)?.squeeze(0)?.squeeze(0)?,
            Gemma::Quantized(model_weights) => model_weights.forward(input_ids, pos)?.squeeze(0)?,
        })
    }
}
