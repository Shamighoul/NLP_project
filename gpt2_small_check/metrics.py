from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import evaluate
import json
import re
from tqdm import tqdm


def evaluate_saved_model(model_dir, VAL_FILE, MAX_NEW_TOKENS=150):
    # Read validation blocks
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        blocks = f.read().strip().split("\n\n")
    # Load model & tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.eval()
    if torch.cuda.is_available(): model.to("cuda")
    # Metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    preds, refs = [], []
    for block in tqdm(blocks):
        lines = block.strip().split("\n")
        # Parse NER list
        ner_lines = [l for l in lines if l.startswith('- ')]
        if ner_lines and ner_lines[0].lstrip().startswith('['):
            elems = [l.lstrip('- ').strip() for l in ner_lines]
            ner_json = ' '.join(elems)
            ner_list = json.loads(ner_json)
        else:
            ner_list = [l.replace('- ', '').strip() for l in ner_lines]
        # Parse reference directions
        dir_lines = [l for l in lines if re.match(r'^\d+\.', l)]
        if dir_lines and dir_lines[0].split(' ',1)[1].startswith('['):
            dir_json = dir_lines[0].split(' ',1)[1]
            dir_list = json.loads(dir_json)
        else:
            dir_list = [l.split('.',1)[1].strip() for l in dir_lines]
        reference = ' '.join(dir_list)
        # Generate
        prompt = "NER:\n" + "\n".join(f"- {e}" for e in ner_list) + "\n\nDirections:\n"
        enc = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            enc = {k: v.to("cuda") for k,v in enc.items()}
        out = model.generate(
            input_ids=enc['input_ids'],
            attention_mask=enc.get('attention_mask', None),
            max_length=enc['input_ids'].shape[-1] + MAX_NEW_TOKENS,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
        preds.append(gen)
        refs.append(reference)
    # Compute and print
    bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])
    rouge_res = rouge.compute(predictions=preds, references=refs)
    print(f"BLEU: {bleu_res['bleu']:.4f}")
    print(f"ROUGE-1: {rouge_res['rouge1']:.4f}, ROUGE-2: {rouge_res['rouge2']:.4f}, ROUGE-L: {rouge_res['rougeL']:.4f}")


evaluate_saved_model('gpt2-ner2directions', 'val_ner2dir.txt')
# import torch
# import evaluate

# bleu = evaluate.load("bleu")
# rouge = evaluate.load("rouge")

# predictions, references = [], []
# for example in val_texts:
#     # parse NER list and reference directions
#     parts = example.strip().split("\n\n")
#     ner_block = parts[0].split("\n")[1:]
#     ref_block = parts[1].split("\n")[1:]
#     ner_list = [line.replace("- ", "").strip() for line in ner_block]
#     ref_text = " ".join([line.split('.',1)[1].strip() for line in ref_block])
#     # generate
#     prompt = "NER:\n" + "\n".join(f"- {e}" for e in ner_list) + "\n\nDirections:\n"
#     encoding = tokenizer(prompt, return_tensors="pt")
#     if torch.cuda.is_available():
#         model.to("cuda")
#         encoding = {k: v.to("cuda") for k, v in encoding.items()}
#     output_ids = model.generate(
#         encoding['input_ids'], attention_mask=encoding['attention_mask'],
#         max_length=encoding['input_ids'].shape[-1] + 150,
#         num_beams=5, no_repeat_ngram_size=2, early_stopping=True,
#         pad_token_id=tokenizer.pad_token_id
#     )
#     gen = tokenizer.decode(output_ids[0], skip_special_tokens=True)[len(prompt):].strip()
#     predictions.append(gen)
#     references.append(ref_text)

# # Compute metrics
# bleu_res = bleu.compute(predictions=predictions, references=[[r] for r in references])
# rouge_res = rouge.compute(predictions=predictions, references=references)
# print("\nBleu Score:", bleu_res['bleu'])
# print("ROUGE-1:", rouge_res['rouge1'], "ROUGE-2:", rouge_res['rouge2'], "ROUGE-L:", rouge_res['rougeL'])