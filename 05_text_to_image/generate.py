"""Generate images from text with SDXL-Turbo, a distilled diffusion model.

Ordinary diffusion models need 25-50 denoising steps per image; SDXL-Turbo was
distilled to produce a 512x512 image in 1-4 steps, so a GPU slice makes several
images per second.

    python generate.py --prompt "a lighthouse on the prairie, oil painting"
    python generate.py --prompts-file prompts.txt --num-images 4 --steps 2
"""
import argparse
import re
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import make_image_grid
from huggingface_hub import snapshot_download

MODEL_FILES = ["model_index.json", "*/config.json", "scheduler/*", "tokenizer*/*", "*/*.fp16.safetensors"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", action="append", help="a prompt (repeatable); default: prompts.txt")
    p.add_argument("--prompts-file", default=Path(__file__).with_name("prompts.txt"))
    p.add_argument("--line", type=int, help="use only this line (0-based) of the prompts file")
    p.add_argument("--num-images", type=int, default=4, help="images per prompt (generated as one batch)")
    p.add_argument("--steps", type=int, default=1, help="denoising steps, 1-4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="outputs/images")
    args = p.parse_args()

    prompts = args.prompt or [l.strip() for l in Path(args.prompts_file).read_text().splitlines() if l.strip()]
    if args.line is not None:
        prompts = [prompts[args.line]]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    # Fetch (or find in the cache) only the fp16 weights: 7 GB instead of the full 40 GB repository
    model_dir = snapshot_download("stabilityai/sdxl-turbo", allow_patterns=MODEL_FILES)
    pipe = AutoPipelineForText2Image.from_pretrained(model_dir, torch_dtype=torch.float16,
                                                     variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True)
    # The VAE turns latents into pixels in fp32; decoding the batch one image at a time
    # keeps that peak small enough for a 10 GB GPU slice.
    pipe.vae.enable_slicing()
    print(f"Pipeline loaded in {time.perf_counter() - t0:.0f}s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB of GPU memory")

    pipe(prompt="warm-up", num_inference_steps=1, guidance_scale=0.0)  # first call pays one-off setup costs
    for i, prompt in enumerate(prompts):
        generator = torch.Generator("cuda").manual_seed(args.seed + i)
        torch.cuda.synchronize()
        start = time.perf_counter()
        images = pipe(prompt=prompt, num_inference_steps=args.steps, guidance_scale=0.0,
                      num_images_per_prompt=args.num_images, generator=generator).images
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start

        slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:50]
        name = f"{args.seed + i:03d}-{slug}"
        for j, image in enumerate(images):
            image.save(out_dir / f"{name}-{j}.png")
        make_image_grid(images, rows=1, cols=len(images)).save(out_dir / f"{name}-grid.png")
        print(f"{len(images)} images in {seconds:.2f}s ({len(images) / seconds:.1f} images/s): {prompt}")

    print(f"Peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB. Images in {out_dir}")


if __name__ == "__main__":
    main()
