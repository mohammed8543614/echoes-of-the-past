import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = M2M100Tokenizer.from_pretrained("mattiadc/hiero-transformer")
model = M2M100ForConditionalGeneration.from_pretrained("mattiadc/hiero-transformer").to(device)
tokenizer.src_lang = "ar"

def score_sequence(codes: list) -> float:
    text = " ".join(codes)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    target_text = "a"
    targets = tokenizer(target_text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        loss = model(**inputs, labels=targets).loss
    return loss.item()

def translate(codes: list) -> str:
    if not codes: return ""
    text = " ".join(codes)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id("en"),
            num_beams=5,
            max_length=128,
            early_stopping=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)