from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import evaluate
import json
import re
from tqdm import tqdm


def evaluate_saved_model(model_dir, VAL_FILE, MAX_NEW_TOKENS=150, BATCH_SIZE=8, METRIC_FREQ=1000):
    # Read validation blocks
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        blocks = f.read().strip().split("\n\n")

    # Parse all blocks into prompts and references
    prompts, refs = [], []
    for block in blocks:
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
        if dir_lines and dir_lines[0].split(' ', 1)[1].startswith('['):
            dir_json = dir_lines[0].split(' ', 1)[1]
            dir_list = json.loads(dir_json)
        else:
            dir_list = [l.split('.', 1)[1].strip() for l in dir_lines]
        reference = ' '.join(dir_list)

        # Build prompt
        prompt = "NER:\n" + "\n".join(f"- {e}" for e in ner_list) + "\n\nDirections:\n"
        prompts.append(prompt)
        refs.append(reference)

    # Load model & tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # correct for decoder-only models

    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

    # Metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    preds = []

    # Batch generation
    total_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in tqdm(range(total_batches), desc="Generating batches"):
        start = batch_idx * BATCH_SIZE
        end = start + BATCH_SIZE
        batch_prompts = prompts[start:end]

        # Tokenize batch with padding
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            enc = {k: v.to("cuda") for k, v in enc.items()}

        # Generate
        with torch.no_grad():
            outs = model.generate(
                input_ids=enc['input_ids'],
                attention_mask=enc.get('attention_mask'),
                max_length=enc['input_ids'].shape[-1] + MAX_NEW_TOKENS,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode and strip prompts
        decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
        for prompt, full_text in zip(batch_prompts, decoded):
            gen = full_text[len(prompt):].strip()
            preds.append(gen)

        # Compute intermediate metrics every METRIC_FREQ batches
        if (batch_idx + 1) % METRIC_FREQ == 0 or (batch_idx + 1) == total_batches:
            count = len(preds)
            refs_so_far = refs[:count]
            bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs_so_far])
            rouge_res = rouge.compute(predictions=preds, references=refs_so_far)
            print(f"After {batch_idx+1} batches ({count} examples): "
                  f"BLEU: {bleu_res['bleu']:.4f}, "
                  f"ROUGE-1: {rouge_res['rouge1']:.4f}, "
                  f"ROUGE-2: {rouge_res['rouge2']:.4f}, "
                  f"ROUGE-L: {rouge_res['rougeL']:.4f}")

    # Final metrics (already printed if last batch)
    if total_batches % METRIC_FREQ != 0:
        # if final metrics weren't printed in loop
        count = len(preds)
        refs_so_far = refs[:count]
        bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs_so_far])
        rouge_res = rouge.compute(predictions=preds, references=refs_so_far)
        print(f"Final - BLEU: {bleu_res['bleu']:.4f}")
        print(f"Final - ROUGE-1: {rouge_res['rouge1']:.4f}, ROUGE-2: {rouge_res['rouge2']:.4f}, ROUGE-L: {rouge_res['rougeL']:.4f}")

evaluate_saved_model(
    'gpt2-ner2directions-optimized',
    'val_ner2dir.txt',
    MAX_NEW_TOKENS=150,
    BATCH_SIZE=8,
    METRIC_FREQ=1
)
