import sys

from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("RufusRubin777/GOT-OCR2_0_CPU", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "RufusRubin777/GOT-OCR2_0_CPU",
    trust_remote_code=True, use_safetensors=True, pad_token_id=tokenizer.eos_token_id,
)
model = model.eval()

text = model.chat(tokenizer, sys.argv[1], ocr_type="ocr")
print(text)
