# 05 · Text-to-image generation with SDXL-Turbo

Turn text prompts into images with a diffusion model. SDXL-Turbo
(Stability AI) was *distilled* to create a 512×512 image in a single denoising
step, where classic diffusion models need 25-50, so a GPU slice produces
several images per second.

**You will learn:** running a generative model pipeline (`diffusers`), GPU memory
limits in practice (out-of-memory errors and how to avoid them), and
"embarrassingly parallel" work as a **job array**.

## Run it

```bash
cd 05_text_to_image
sbatch job.sh                                          # 4 images for each line of prompts.txt
sbatch job.sh --prompt "a moose in a spacesuit, photo" --num-images 8
```

```
Pipeline loaded in 2s, 6.6 GiB of GPU memory
4 images in 1.54s (2.6 images/s): a bison made of glowing data streams, digital art
...
Peak GPU memory 7.7 GiB. Images in outputs/<job-id>
```

To look at the images: open them in VS Code (Remote-SSH), or copy them to your
laptop with `scp -r <user>@<plato login node>:hpcw/05_text_to_image/outputs/<job-id> .`
Each prompt also gets a `...-grid.png` with all its images side by side.

## Job arrays: one task per prompt

[job_array.sh](job_array.sh) submits 8 tasks. Task *N* renders line *N* of
[prompts.txt](prompts.txt), and all of them write to the same directory:

```bash
sbatch job_array.sh
squeue --me                       # sdxl-array_[4-7] waiting, 0-3 running (%4 = max 4 at once)
ls outputs/array-<array job id>
```

This is the pattern for any "same program, many inputs" work: rendering frames,
processing files, running simulations with different parameters.

## Things to try

1. **Steps:** `--steps 1`, `2` and `4`. Does quality improve? How does time scale?
2. **Out of memory:** raise `--num-images` until the job fails with
   `torch.OutOfMemoryError`. Then read `generate.py`: what does
   `pipe.vae.enable_slicing()` do, and why does removing it make even 4 images fail
   on a 10 GB slice?
3. **Seeds:** the same prompt and `--seed` give the same image. Make 3 variations of
   your favourite prompt by changing only the seed.
4. **Bigger GPU:** on a full A100 (`sbatch --gres=gpu:a100:1 job.sh --num-images 16`),
   how many images per second do you get?
5. **Your own prompts:** edit `prompts.txt` (one prompt per line) and change
   `--array=0-7` in `job_array.sh` to match the number of lines.
