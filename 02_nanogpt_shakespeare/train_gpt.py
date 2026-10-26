"""Train a small GPT from scratch on Shakespeare, then let it write some "Shakespeare".

A compact version of Andrej Karpathy's nanoGPT (github.com/karpathy/nanoGPT): the
same architecture family as GPT-2 and today's LLMs, just tiny and character-level
(the vocabulary is the 65 characters that appear in the text).

    python train_gpt.py                              # train with defaults, then sample
    python train_gpt.py --n-layer 6 --n-embd 384     # bigger model
    python train_gpt.py --sample-only --prompt "ROMEO:"
"""
import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.dropout = cfg.n_head, cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # Fused attention kernel (FlashAttention on recent GPUs): never materializes the T x T matrix
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.resid_drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.n_embd), nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False), nn.GELU(),
                                 nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying, as in GPT-2
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.cfg.block_size:])[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return idx


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.path.join(os.environ.get("HPCW_DATA", "."), "shakespeare.txt"))
    p.add_argument("--out-dir", default="outputs/nanogpt")
    # model size
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--block-size", type=int, default=256, help="context length in characters")
    p.add_argument("--dropout", type=float, default=0.1)
    # training
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--compile", action="store_true", help="compile the model with torch.compile")
    # sampling
    p.add_argument("--sample-only", action="store_true", help="skip training, sample from the saved model")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--num-chars", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()

    torch.manual_seed(1337)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Character-level "tokenizer": every distinct character is a token
    text = Path(args.data).read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(chars[i] for i in ids)
    data = torch.tensor(encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split].to(device), data[split:].to(device)  # 1 MB: just keep it on the GPU

    def get_batch(source):
        ix = torch.randint(len(source) - args.block_size - 1, (args.batch_size,), device=device)
        offsets = torch.arange(args.block_size + 1, device=device)
        chunk = source[ix[:, None] + offsets]
        return chunk[:, :-1], chunk[:, 1:]  # targets are the inputs shifted by one character

    args.vocab_size = len(chars)
    ckpt_path = out_dir / "model.pt"
    if args.sample_only:
        ckpt = torch.load(ckpt_path, map_location=device)
        model = GPT(argparse.Namespace(**ckpt["config"])).to(device)
        model.load_state_dict(ckpt["model"])
    else:
        model = GPT(args).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"GPT with {n_params / 1e6:.2f}M parameters, vocabulary of {len(chars)} characters, "
              f"{len(train_data) / 1e6:.2f}M training characters")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                                      weight_decay=0.1, fused=True)
        train_model = torch.compile(model) if args.compile else model
        warmup = args.iters // 20

        def lr_at(it):  # linear warm-up, then cosine decay to 10% of the peak
            if it < warmup:
                return args.lr * (it + 1) / warmup
            progress = (it - warmup) / max(1, args.iters - warmup)
            return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

        @torch.no_grad()
        def estimate_loss():
            train_model.eval()
            losses = {}
            for name, source in (("train", train_data), ("val", val_data)):
                with autocast:
                    losses[name] = torch.stack([train_model(*get_batch(source)) for _ in range(20)]).mean().item()
            train_model.train()
            return losses

        tokens_per_iter = args.batch_size * args.block_size
        best_val, best_iter = float("inf"), 0
        t0 = time.perf_counter()
        for it in range(args.iters + 1):
            if it % args.eval_every == 0 or it == args.iters:
                losses = estimate_loss()
                elapsed = time.perf_counter() - t0
                rate = f"{it * tokens_per_iter / elapsed:9,.0f} tokens/s" if it else ""
                # The data set is small, so the model eventually overfits: keep only the best checkpoint
                improved = losses["val"] < best_val
                if improved:
                    best_val, best_iter = losses["val"], it
                    torch.save({"model": model.state_dict(), "config": vars(args)}, ckpt_path)
                print(f"iter {it:5d}  train loss {losses['train']:.3f}  val loss {losses['val']:.3f}"
                      f"{' *' if improved else '  '}  {elapsed:6.1f}s  {rate}", flush=True)
            if it == args.iters:
                break
            for group in optimizer.param_groups:
                group["lr"] = lr_at(it)
            x, y = get_batch(train_data)
            with autocast:
                loss = train_model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        print(f"Best val loss {best_val:.3f} at iter {best_iter} (marked *), saved to {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])

    model.eval()
    idx = torch.tensor([encode(args.prompt)], device=device)
    with autocast:
        out = model.generate(idx, args.num_chars, temperature=args.temperature)
    print("\n" + "-" * 30 + " generated text " + "-" * 30)
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
