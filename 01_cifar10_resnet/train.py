"""Train a ResNet on CIFAR-10 (60k tiny 32x32 images, 10 classes).

The classic "hello world" of deep learning, written as a cluster job should be:
  * reads the data from one file copied to node-local disk ($SLURM_TMPDIR)
  * saves a checkpoint every epoch and resumes from it when restarted
  * reports throughput (images/s) so you can see what the GPU is waiting for
  * writes results.json so job-array sweeps can be compared afterwards

Models: ResNet-9 (the fast DAWNBench-winning design, ~6.5M parameters) or a
CIFAR-sized ResNet-18 (~11M parameters).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1) * 255


# ----------------------------------------------------------------------------- models
def conv_bn(c_in, c_out, pool=False):
    layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(conv_bn(channels, channels), conv_bn(channels, channels))

    def forward(self, x):
        return x + self.block(x)


class Scale(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor


def resnet9(num_classes=10):
    return nn.Sequential(
        conv_bn(3, 64),
        conv_bn(64, 128, pool=True), Residual(128),
        conv_bn(128, 256, pool=True),
        conv_bn(256, 512, pool=True), Residual(512),
        nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(512, num_classes, bias=False),
        Scale(0.125),  # keeps the initial logits small, which makes training much more stable
    )


def resnet18(num_classes=10):
    model = torchvision.models.resnet18(num_classes=num_classes)
    # ImageNet ResNets shrink the input 4x straight away; CIFAR images are only 32x32.
    model.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


MODELS = {"resnet9": resnet9, "resnet18": resnet18}


# ----------------------------------------------------------------------------- data
class CifarTrain(Dataset):
    """Augmentation on the CPU, in DataLoader worker processes (the usual PyTorch way)."""

    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2)  # NHWC -> NCHW, uint8
        self.labels = torch.from_numpy(labels)
        self.augment = v2.Compose([v2.RandomCrop(32, padding=4, padding_mode="reflect"),
                                   v2.RandomHorizontalFlip()])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.augment(self.images[i]), self.labels[i]


def gpu_batches(padded, labels, batch_size, device):
    """The same augmentation done on the GPU, a whole batch at a time (no CPU workers needed).
    `padded` holds all training images, already padded by 4 pixels on each side."""
    n = len(labels)
    perm = torch.randperm(n, device=device)
    for start in range(0, n - batch_size + 1, batch_size):
        idx = perm[start:start + batch_size]
        dx, dy = np.random.randint(0, 9, size=2)  # one random crop offset per batch: cheap and good enough
        x = padded[idx, :, dy:dy + 32, dx:dx + 32]
        flip = torch.rand(batch_size, device=device) < 0.5
        x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
        yield x, labels[idx]


def normalize(x, device):
    return ((x.to(device, non_blocking=True).float() - MEAN.to(device)) / STD.to(device)).contiguous(
        memory_format=torch.channels_last)


# ----------------------------------------------------------------------------- training
@torch.no_grad()
def evaluate(model, x_test, y_test, device, amp):
    model.eval()
    correct = 0
    for i in range(0, len(y_test), 1000):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(normalize(x_test[i:i + 1000], device))
        correct += (logits.argmax(1) == y_test[i:i + 1000].to(device)).sum().item()
    return correct / len(y_test)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.path.join(os.environ.get("HPCW_DATA", "."), "cifar10.npz"))
    p.add_argument("--out-dir", default="outputs/run")
    p.add_argument("--model", choices=MODELS, default="resnet9")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.2, help="peak learning rate (one-cycle schedule)")
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--compile", action="store_true", help="compile the model with torch.compile")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 2)),
                   help="DataLoader worker processes (default: the job's CPU count)")
    p.add_argument("--gpu-aug", action="store_true", help="keep the data on the GPU and augment there")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    amp = args.precision == "bf16"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    data = np.load(args.data)
    x_test = torch.from_numpy(data["x_test"]).permute(0, 3, 1, 2)
    y_test = torch.from_numpy(data["y_test"])
    if args.gpu_aug:
        padded = np.pad(data["x_train"], ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")
        x_train = torch.from_numpy(padded).permute(0, 3, 1, 2).to(device)
        y_train = torch.from_numpy(data["y_train"]).to(device)
        steps_per_epoch = len(y_train) // args.batch_size
    else:
        loader = DataLoader(CifarTrain(data["x_train"], data["y_train"]), batch_size=args.batch_size,
                            shuffle=True, drop_last=True, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0)
        steps_per_epoch = len(loader)

    # Model, optimizer, learning-rate schedule
    model = MODELS[args.model]().to(device, memory_format=torch.channels_last)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, pct_start=0.25,
                                                    total_steps=args.epochs * steps_per_epoch)
    train_model = torch.compile(model) if args.compile else model

    # Resume from the last checkpoint if this run was interrupted
    ckpt_path = out_dir / "checkpoint.pt"
    start_epoch = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        if ckpt["model_name"] != args.model:
            raise SystemExit(f"{ckpt_path} holds a {ckpt['model_name']} checkpoint, not {args.model}. "
                             "Use a different --out-dir (or job name) for a new run.")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {ckpt_path} at epoch {start_epoch}")

    total_train_time = 0.0
    for epoch in range(start_epoch, args.epochs):
        train_model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        batches = (gpu_batches(x_train, y_train, args.batch_size, device) if args.gpu_aug else loader)
        for x, y in batches:
            x, y = normalize(x, device), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = F.cross_entropy(train_model(x), y, label_smoothing=0.1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.detach()  # no .item() here: it would make the CPU wait for the GPU every step
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0
        total_train_time += epoch_time

        acc = evaluate(model, x_test, y_test, device, amp)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {loss_sum.item() / steps_per_epoch:.3f}  "
              f"test acc {acc:6.2%}  {epoch_time:5.1f}s  "
              f"{steps_per_epoch * args.batch_size / epoch_time:7.0f} img/s", flush=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "model_name": args.model}, ckpt_path)

    if start_epoch >= args.epochs:
        acc = evaluate(model, x_test, y_test, device, amp)
    results = {**vars(args), "test_acc": acc, "train_seconds": round(total_train_time, 1),
               "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "gpu": torch.cuda.get_device_name(0), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Final test accuracy {acc:.2%}, training time {total_train_time:.0f}s, "
          f"peak GPU memory {results['peak_gpu_mem_gib']} GiB. Results in {out_dir}")


if __name__ == "__main__":
    main()
