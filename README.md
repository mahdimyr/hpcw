# AI and Machine Learning on Plato: hands-on examples

Six small, complete examples of current AI workloads, written to run as Slurm
jobs on Plato's GPUs. Each finishes in a few minutes on a 1/7 slice of an
NVIDIA A100, so a whole class can run them at the same time.

| | Example | What it does | Main lessons | Time* |
|---|---|---|---|---|
| 00 | [GPU hello](00_gpu_hello/) | checks your GPU, CPU vs. GPU speed | first job, GPU requests, fp32/tf32/bf16 | < 1 min |
| 01 | [CIFAR-10 ResNet](01_cifar10_resnet/) | image classifier, 91.6% accuracy | data staging, checkpoint/restart, **job arrays**, data loading | 1.5 min |
| 02 | [nanoGPT](02_nanogpt_shakespeare/) | trains a GPT from scratch on Shakespeare | transformer training, overfitting, throughput | 1.5 min |
| 03 | [LLM inference](03_llm_inference/) | Qwen3 / Qwen3.5 chat, vLLM batch inference, your own LLM API server | model memory, serving engines, services as jobs | 1-4 min |
| 04 | [LoRA fine-tuning](04_llm_finetune_lora/) | teaches an LLM text-to-SQL: 3.5% → 74.5% | parameter-efficient fine-tuning, HF training stack | 7 min |
| 05 | [Text-to-image](05_text_to_image/) | SDXL-Turbo, 2.6 images/s | generative pipelines, GPU out-of-memory, job arrays | 1 min |

\*on a `1g.10gb` GPU slice, measured October 2026. Start with 00; the others can be done in any order.

## Quick start

Log in to Plato, then:

```bash
cp -r <workshop folder> ~/hpcw          # or: git clone <repository URL> ~/hpcw
cd ~/hpcw/00_gpu_hello
sbatch job.sh                           # submit the job
squeue --me                             # watch it: PD = pending, R = running
cat gpu-hello-*.out                     # read its output when it's done
```

Every example works the same way: `cd` into its folder, `sbatch` its job script,
read the `.out` file. Its README explains what happened and suggests experiments.

**Educational (`hpc_s_...`) accounts:** run `export SBATCH_GRES=gpu:a100:1` first
(see [GPU types](#asking-for-a-gpu) below).

---

## Plato essentials

### Asking for a GPU

Plato routes your job to the right partition from your **account**: never pass
`--partition` (that is an error), and always give a **GPU type**:

```bash
#SBATCH --gres=gpu:1g.10gb:1     # type:count
```

| GPU type | What it is | Memory | Where | Available to |
|---|---|---|---|---|
| `a100` | a full NVIDIA A100 | 80 GB | platogpu001-002 (1 each) | all accounts |
| `3g.40gb` | 3/7 of an A100 (MIG slice) | 40 GB | platogpu003 (1) | research (`hpc_p_...`) accounts |
| `2g.20gb` | 2/7 of an A100 | 20 GB | platogpu003 (2) | research accounts |
| `1g.10gb` | 1/7 of an A100 | 10 GB | platogpu004 (7) | research accounts |

A **MIG** (Multi-Instance GPU) slice is a hardware partition of an A100 with its
own memory and compute units: a small but complete GPU. The examples request
`1g.10gb` slices because there are seven of them per GPU and every example fits.

To use another type, override the script's `--gres` line:

```bash
sbatch --gres=gpu:a100:1 job.sh          # for one job
export SBATCH_GRES=gpu:a100:1            # for every sbatch in this login session
```

**Ask only for what you use.** platogpu004 has 16 CPU cores and 251 GB of RAM for
its 7 slices, so a slice job asking for more than 2 cores or ~32 GB prevents others
from using the remaining slices. Always set `--mem`: the default is 512 MB per core.

### Everyday commands

| Command | What it does |
|---|---|
| `sbatch job.sh [args]` | submit a batch job (extra args go to the script) |
| `squeue --me` | your pending and running jobs (`--start` estimates start times) |
| `scancel <id>` | cancel a job (`scancel -u $USER` cancels all of yours) |
| `salloc --gres=gpu:1g.10gb:1 -c 2 --mem=8G -t 30` | interactive shell on a GPU node (`exit` when done!) |
| `srun --jobid=<id> --overlap nvidia-smi` | run a command inside your running job, e.g. to check its GPU memory |
| `sacct -j <id> --format=JobID,Elapsed,State,MaxRSS,AllocTRES%50` | what a finished job used |
| `seff <id>` | efficiency summary: CPU and memory used vs. requested |

Note: `nvidia-smi` shows memory but no utilization percentage for MIG slices.
The examples print their own throughput (images/s, tokens/s) instead.

### Where to put files

| Location | Size | Use it for |
|---|---|---|
| `/home/$USER` | small quota, backed up | code, scripts, results |
| `/scratch/$USER` | large, **not** backed up | environments, datasets, large checkpoints, model caches |
| `$SLURM_TMPDIR` | ~200 GB local NVMe per GPU node, deleted when the job ends | fast scratch space during a job ([example](01_cifar10_resnet/job.sh)) |

Each example writes its results to an `outputs/` folder next to its job script
(about 400 MB for one pass through all of them). Caches and your own environments
go to `/scratch/$USER/hpcw` (`$HPCW_WORK`). In your own projects, put checkpoints of
several GB on `/scratch`, e.g. `sbatch job.sh --out-dir /scratch/$USER/my-run`.

---

## Software environment

Every job script starts with `source ../setup/env.sh`. [setup/env.sh](setup/env.sh):

1. loads the software modules: `StdEnv/2023`, `python/3.12` and `arrow` (for
   `datasets`), plus `opencv` for vLLM ([setup/modules.sh](setup/modules.sh));
2. activates a Python virtual environment, by default the one prepared for the
   workshop;
3. points all caches (Hugging Face, Triton, torch.compile, vLLM) at `/scratch`.

Plato uses the **Alliance (Compute Canada) software stack**: thousands of
modules (`module avail`, `module spider <name>`) and a "wheelhouse" of Python
packages pre-built for the cluster. You install from it with `pip install --no-index`.
It needs no internet and compiles nothing. `avail_wheels torch` lists what is there.

There are two environments, because vLLM requires its own PyTorch version:

| Environment | Activate with | Contents ([pinned](setup/requirements-main.txt)) |
|---|---|---|
| main | `source setup/env.sh` | PyTorch 2.14, transformers 5.20, peft, trl, datasets, diffusers |
| vllm | `source setup/env.sh vllm` | vLLM 0.25 (with PyTorch 2.11) |

### Exercise: build your own environment

```bash
setup/build_env.sh main                        # ~1 minute, into /scratch/$USER/hpcw/venv
export HPCW_VENV=/scratch/$USER/hpcw/venv      # env.sh now uses yours
source setup/env.sh
avail_wheels scikit-learn                      # is it in the wheelhouse?
pip install --no-index scikit-learn            # add a package
```

Pin your versions in a requirements file ([example](setup/requirements-main.txt)):
the environment can then be rebuilt identically, which is what makes results
reproducible. Watch for silent conflicts. Installing `causal_conv1d` from the
wheelhouse, for example, quietly downgrades PyTorch to 2.9 and breaks torchvision.
`pip check` catches this.

### Models and datasets

All models and datasets are pre-downloaded to the shared area, and jobs run with
`HF_HUB_OFFLINE=1`. Thirty people don't download the same 30 GB at once, and
jobs don't depend on the internet.

**Bring your own model.** Download it on the login node into your own folder,
then pass the folder instead of the model name:

```bash
source setup/env.sh
HF_HUB_OFFLINE=0 hf download HuggingFaceTB/SmolLM3-3B --local-dir $HPCW_WORK/models/SmolLM3-3B
sbatch job_chat.sh --model $HPCW_WORK/models/SmolLM3-3B      # in 03_llm_inference
```

Gated models (Llama, Gemma) need a Hugging Face account, accepting the license on
the model's page, and `hf auth login` once.

---

## Troubleshooting

| Message | Meaning and fix |
|---|---|
| `You are not allowed to submit to partition` | remove `-p`/`--partition`: Plato picks the partition |
| `GPU requests must include the type` | use `--gres=gpu:<type>:1`, e.g. `gpu:a100:1` |
| `Requested node configuration is not available` | that GPU type doesn't exist for your account: `export SBATCH_GRES=gpu:a100:1` |
| job stays `PD` with `(Resources)` or `(Priority)` | normal: all suitable GPUs are busy. Shorter `--time` and smaller requests start sooner |
| `torch.OutOfMemoryError: CUDA out of memory` | GPU memory is full: smaller batch, smaller model, or a bigger GPU type |
| `State: OUT_OF_MEMORY` in `sacct` | host RAM, not GPU: raise `--mem` |
| `State: TIMEOUT` | raise `--time`, or resume from a checkpoint (example 01) |
| `env.sh: no Python environment at ...` | the shared area isn't readable, or build your own (above) |
| `OfflineModeIsEnabled` / `not cached locally` | the model isn't pre-downloaded: see "Bring your own model" |
| `../setup/env.sh: No such file or directory` | submit from inside the example's folder |

---

## For instructors

**One-time setup** (login node, ~10 minutes, ~35 GB in `/scratch`):

```bash
setup/instructor_setup.sh /scratch/$USER/hpcw-shared
```

This builds both environments, downloads the models and datasets
([setup/prefetch.sh](setup/prefetch.sh)) and makes the directory world-readable.
If you choose another location, change `HPCW_SHARED` in [setup/env.sh](setup/env.sh).

**Before the workshop:**

- Make sure attendees can read the shared area: log in as a test attendee and run
  `source setup/env.sh && python -c "import torch"`.
- Run each example once yourself. `vllm`, `torch.compile` and Qwen3.5 compile GPU
  kernels on first use, and these caches are per user (`$HPCW_WORK/cache`), so
  attendees' first runs of example 03 take 2-3 minutes longer than later ones.
- Check `/scratch`'s purge policy, so the shared area isn't deleted before the day.
- **GPU capacity.** Educational (`hpc_s_...`) accounts can use only the two full A100s
  (partition `plato_gpu_edu`), so a class of 20 would mostly wait in the queue.
  Ask the cluster administrators for a reservation for the session (`--reservation=`),
  and ideally for the A100s to be split into MIG slices (14 × `1g.10gb`), or for
  access to the `1g.10gb` slices on platogpu004.
