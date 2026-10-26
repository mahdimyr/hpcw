# 02 · Train a GPT from scratch (nanoGPT)

Train a small GPT language model on the complete works of Shakespeare (1 MB of
text), then let it write new "Shakespeare". It has the same architecture as
GPT-2 and today's LLMs, just tiny: 3 million parameters instead of billions, and
characters instead of word pieces. Based on Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT) and his video
[Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY).

**You will learn:** what training a transformer looks like, train vs. validation
loss (overfitting), tokens/s as a throughput metric, how model size drives GPU
memory and time, and fused attention kernels (FlashAttention).

## Run it

```bash
cd 02_nanogpt_shakespeare
sbatch job.sh                  # 2000 iterations, ~1.5 minutes on a 1g.10gb slice
```

```
GPT with 3.23M parameters, vocabulary of 65 characters, 1.00M training characters
iter     0  train loss 4.161  val loss 4.160 *     0.7s
iter   500  train loss 1.613  val loss 1.802 *    21.4s    383,258 tokens/s
iter  1000  train loss 1.323  val loss 1.543 *    42.0s    390,315 tokens/s
iter  2000  train loss 1.174  val loss 1.455 *    83.2s    393,943 tokens/s
Best val loss 1.455 at iter 2000 (marked *), saved to outputs/nanogpt/model.pt

------------------------------ generated text ------------------------------
GLOUCESTER:
A word, thou takest me of both, my door; these eyes
Be all deserved in despised for the earth:
```

A loss of 4.17 = ln(65) means random guessing among 65 characters. The model
learns spelling, then words, then the play format (`NAME:` + verse).

Sample more text from the trained model, with your own opening:

```bash
salloc --gres=gpu:1g.10gb:1 --cpus-per-task=2 --mem=8G --time=00:15:00
source ../setup/env.sh
python train_gpt.py --sample-only --out-dir outputs/nanogpt --prompt "JULIET:" --temperature 1.0
```

## Things to try

1. **Scale up:** `sbatch --job-name=big job.sh --n-layer 6 --n-head 6 --n-embd 384 --iters 3000`
   (11M parameters, the nanoGPT reference configuration, ~5 minutes on a slice). How
   do tokens/s, peak GPU memory and val loss change?
2. **Overfitting:** 1 MB of text is not much for 11M parameters. With the bigger
   model, train loss keeps falling, but val loss stops improving and then rises:
   the model memorizes instead of generalizing. The checkpoint is only saved when val
   loss improves (`*`). Try `--dropout 0` and `--dropout 0.3`: which overfits sooner?
3. **Context length:** `--block-size 512` doubles the context the model sees. What
   happens to memory and speed? Attention cost grows with the square of the context.
4. **Precision and compilation:** compare `--precision fp32` and `--compile` with
   the default. Which gives the biggest speed-up here?
5. **Temperature:** sample with `--temperature 0.5` and `1.2`. What changes?
6. **Full GPU:** on an A100, the reference configuration
   (`sbatch --gres=gpu:a100:1 job.sh --n-layer 6 --n-head 6 --n-embd 384 --dropout 0.2 --iters 5000 --compile`)
   runs at ~900,000 tokens/s and finishes in 90 seconds. Its best val loss, 1.47 around
   iteration 1500, matches Karpathy's published result (1.4697).
