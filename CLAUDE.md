# CLAUDE.md

Semi-supervised CLIP adaptation on RSICD (course assignment). Full task spec: `docs/TASK.md`. Progress log: `docs/PROGRESS.md`. At the start of every session, read both before doing anything.

## Hard rules
- Split A: images only. Never read A captions, labels, or class names from filenames.
- Split B: images + captions may be used for training. Class labels never for training.
- Split C: final evaluation only, and only after the user explicitly approves. Never bypass the C lock (`eval.confirm_test`), never tune anything on C.
- The caption bank is built from the B-train part of the current fold only.
- Work only on branch `shipeng/bridge`. Never push to or modify other branches.
- Never commit `data/raw/`, `runs/`, or `*.pt`.

## Running (Slurm cluster + Apptainer)
- You are on a login node. Never run GPU work here. All training, feature extraction and evaluation go through `sbatch scripts/slurm/run.sbatch <command>`, submitted from the repository root.
- The login node is x86_64 but the container is arm64, so the container cannot run on the login node. Plain file operations (git, reading logs and CSVs, editing code) happen on the login node. Anything that needs the container, including unit tests and short checks, runs on a compute node: for short commands use a blocking `srun -A naiss2026-1-47-gpu -p gpu --nodes=1 --gpus-per-node=1 -t 00:10:00 apptainer exec --nv --bind /nobackup/proj/disk/cvls/personal/shipengl/rsicd --env HF_HOME=/nobackup/proj/disk/cvls/personal/shipengl/rsicd/hf_cache /nobackup/proj/disk/cvls/personal/shipengl/rsicd/rsicd.sif <command>`, which prints the output directly; batch several checks into one srun call. Use sbatch for anything longer than a few minutes.
- Slurm logs: `/nobackup/proj/disk/cvls/personal/shipengl/rsicd/logs/<job-name>_<jobid>.out|.err`. Give every job a descriptive name with `-J`.
- Monitor with `squeue -u $USER` and `sacct -j <jobid> --format=JobID,JobName,State,Elapsed,ExitCode`. Do not poll more often than every few minutes; if jobs stay pending for long, tell the user instead of waiting silently.
- The adapter trains on cached features, so single runs are short: bundle related short runs (folds, seeds, grid points) into one job or a job array instead of many tiny jobs.
- Before any submission, estimate its GPU-hours. Ask the user before submitting anything above 10 GPU-hours in total.
- Always pass `model=vit_b32_laion2b device=cuda` (config defaults are ViT-L/14 and mps).
- Development runs use `data=kfold5` and evaluate on the held-out B fold.

## Working style
- Small commits, one change each. Update `docs/PROGRESS.md` after each step.
- Stop and ask the user when: images are missing, reproduction misses its acceptance criteria, a data rule is unclear, results contradict expectations, anything touches split C, or the design in `docs/TASK.md` needs to change.
- Never change pre-registered criteria, grids, or selection rules after seeing results. Report negative results honestly.