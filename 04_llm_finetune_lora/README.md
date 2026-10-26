# 04 · Fine-tune an LLM with LoRA: text to SQL

Teach a small LLM a new skill: turning English questions about a database into
SQL queries. This is how most people adapt open-weight models today.

**You will learn:** parameter-efficient fine-tuning (LoRA), the Hugging Face
training stack (`transformers` + `peft` + `trl`), measuring a model before and
after training, and why fine-tuning fits on a fraction of a GPU.

**LoRA** (Low-Rank Adaptation) freezes the pre-trained model and trains small
extra matrices added to its layers: here 10M trainable parameters next to the
model's 600M frozen ones (1.7%). The result is an "adapter" of a few tens of MB
that you load on top of the original model.

**Data:** [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context),
78,000 (question, `CREATE TABLE` schema, SQL answer) triples. The script trains on
4,000 of them and evaluates on 200 others the model never sees during training.

## Run it

```bash
cd 04_llm_finetune_lora
sbatch job.sh                  # ~7 minutes on a 1g.10gb slice
```

```
Base model exact-match accuracy: 3.5%
trainable params: 10,092,544 || all params: 606,142,464 || trainable%: 1.6650
{'loss': '0.3809', ..., 'mean_token_accuracy': '0.9307', 'epoch': '0.1'}
...
Training took 315s, peak GPU memory 2.50 GiB
Exact-match accuracy: 3.5% before -> 74.5% after fine-tuning

QUESTION:  What is the highest pick number of a defensive back with less than 136 overall points?
SCHEMA:    CREATE TABLE table_name_99 (pick__number INTEGER, overall VARCHAR, position VARCHAR)
REFERENCE: SELECT MAX(pick__number) FROM table_name_99 WHERE overall < 136 AND position = "defensive back"
BEFORE:    SELECT MAX(pick__number) FROM table_name_99 WHERE position = 'defensive' AND overall < 136;
AFTER:     SELECT MAX(pick__number) FROM table_name_99 WHERE position = "defensive back" AND overall < 136
```

The log also shows two harmless warnings: one about PAD/BOS/EOS tokens, and
PEFT's "Could not find a config file" (it can't check the model online in offline mode).

**Read the examples critically.** "Exact match" compares text, so the base model
loses points for `'athens'` vs. `"Athens"` even when its query would work. Part of
what fine-tuning taught it is the dataset's *style*, and part is the skill. How
would you measure "correct SQL" more fairly? (Hint: run both queries on a database.)

## Things to try

1. **Training length:** `--steps 50` vs. `--steps 500`. How quickly does accuracy saturate?
2. **LoRA rank:** `--lora-rank 4` and `--lora-rank 64`. How do trainable parameters,
   memory, time and accuracy change?
3. **Newer model:** `sbatch --time=00:25:00 job.sh --model Qwen/Qwen3.5-0.8B`. Is the
   2026 generation better before fine-tuning? After? It trains about twice as slowly
   (the log warns that an optimized kernel, `causal_conv1d`, is missing): new
   architectures often need new GPU kernels before they are fast.
4. **Memory:** the job prints its peak GPU memory. Increase `--batch-size` until it no
   longer fits in 9.5 GB. What runs out first?
5. **Use the adapter:** load it in your own script with
   `PeftModel.from_pretrained(base_model, "<adapter dir>")` and ask it questions about
   a schema you invent.
