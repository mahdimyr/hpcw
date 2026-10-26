"""Teach a small LLM a new skill with LoRA: turning questions into SQL queries.

LoRA (Low-Rank Adaptation) freezes the pre-trained model and trains small
low-rank matrices added to its layers -- typically <1% of the parameters -- so
fine-tuning fits on a single, even fractional, GPU. This is how most people adapt
open-weight LLMs today.

Steps:
  1. measure the base model on held-out text-to-SQL questions (exact-match accuracy)
  2. fine-tune LoRA adapters with TRL's SFTTrainer
  3. measure again and show some before/after answers

Data: b-mc2/sql-create-context (questions + CREATE TABLE statements -> SQL), 78k examples.
"""
import argparse
import re
import time

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

SYSTEM = "You translate questions into SQL. Reply with a single SQL query and nothing else."


def build_prompt(tokenizer, example):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Schema: {example['context']}\nQuestion: {example['question']}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def normalize_sql(sql):
    sql = re.sub(r"^```(sql)?|```$", "", sql.strip()).strip()  # strip markdown code fences
    return re.sub(r"\s+", " ", sql).rstrip(";").strip().lower()


@torch.no_grad()
def evaluate(model, tokenizer, examples, batch_size=32):
    """Greedy-decode every example and count exact matches with the reference SQL."""
    model.eval()
    tokenizer.padding_side = "left"  # generation continues from the right end of each row
    predictions = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        prompts = [build_prompt(tokenizer, dict(zip(batch, values))) for values in zip(*batch.values())]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        predictions += tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    correct = sum(normalize_sql(p) == normalize_sql(a) for p, a in zip(predictions, examples["answer"]))
    return correct / len(predictions), predictions


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out-dir", default="outputs/lora")
    p.add_argument("--train-examples", type=int, default=4000)
    p.add_argument("--eval-examples", type=int, default=200)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    args = p.parse_args()

    # snapshot_download returns the local copy in the Hugging Face cache (works offline)
    data_dir = snapshot_download("b-mc2/sql-create-context", repo_type="dataset")
    data = load_dataset(data_dir, split="train").shuffle(seed=0)
    eval_set = data.select(range(args.eval_examples))
    train_set = data.select(range(args.eval_examples, args.eval_examples + args.train_examples))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")

    t0 = time.perf_counter()
    acc_before, preds_before = evaluate(model, tokenizer, eval_set)
    print(f"Base model exact-match accuracy: {acc_before:.1%}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    # Prompt-completion format: the loss is computed on the completion (the SQL) only.
    train_set = train_set.map(lambda ex: {"prompt": build_prompt(tokenizer, ex), "completion": ex["answer"]},
                              remove_columns=train_set.column_names)
    lora = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    config = SFTConfig(
        output_dir=args.out_dir, max_steps=args.steps, per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=args.steps // 20,
        bf16=True, max_length=512, logging_steps=25, save_strategy="no", report_to="none",
    )
    tokenizer.padding_side = "right"
    trainer = SFTTrainer(model=model, args=config, train_dataset=train_set, processing_class=tokenizer,
                         peft_config=lora)
    trainer.model.print_trainable_parameters()
    t0 = time.perf_counter()
    trainer.train()
    print(f"Training took {time.perf_counter() - t0:.0f}s, "
          f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    trainer.save_model(args.out_dir)  # saves only the LoRA adapter (a few MB)
    print(f"LoRA adapter saved to {args.out_dir}")

    acc_after, preds_after = evaluate(trainer.model, tokenizer, eval_set)
    print(f"\nExact-match accuracy: {acc_before:.1%} before -> {acc_after:.1%} after fine-tuning\n")
    for i in range(3):
        print(f"QUESTION:  {eval_set[i]['question']}\nSCHEMA:    {eval_set[i]['context']}\n"
              f"REFERENCE: {eval_set[i]['answer']}\nBEFORE:    {preds_before[i].strip()[:300]}\n"
              f"AFTER:     {preds_after[i].strip()}\n")


if __name__ == "__main__":
    main()
