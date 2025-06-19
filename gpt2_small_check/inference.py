from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


def setup_inference(model_dir: str):
    tok = GPT2Tokenizer.from_pretrained(model_dir)
    tok.pad_token = tok.eos_token
    mdl = GPT2LMHeadModel.from_pretrained(model_dir)
    mdl.eval()
    return tok, mdl

def generate_directions(ner_list, model, tokenizer, max_new_tokens=150):
    prompt = "NER:\n" + "\n".join(f"- {ent.strip()}" for ent in ner_list) + "\n\nDirections:\n"
    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    if torch.cuda.is_available():
        model.to("cuda")
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=input_ids.shape[-1] + max_new_tokens,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id
    )
    full = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full[len(prompt):].strip()

tokenizer, model = setup_inference('gpt2-ner2directions')
prediction = generate_directions(
    ["onion", "garlic", "tomatoes", "olive oil", "basil", "salt", "pepper"],
    model,
    tokenizer
)
print(prediction)