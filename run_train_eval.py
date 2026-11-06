#!/usr/bin/env python3
"""
Wrapper for lerobot-train (training) and lerobot-rollout (policy eval on the real arm).

It imports robot / camera / dataset settings from run_record.py,
so recording and eval always use the SAME camera names and resolution.
Keep this file in the same folder as run_record.py.

Tuned for my system : NVIDIA RTX 3050 Laptop GPU (4 GB VRAM), ~16 GB system RAM, WSL.

Usage:
    python run_train_eval.py train                  # train ACT on the local dataset
    python run_train_eval.py train --push           # also push the policy to the Hub
    python run_train_eval.py train --fresh          # delete old training output first
    python run_train_eval.py resume                 # resume from the last checkpoint
    python run_train_eval.py resume --steps 40000   # resume and extend total steps
    python run_train_eval.py list                   # list saved checkpoints
    python run_train_eval.py eval                   # run the last checkpoint on the real arm
    python run_train_eval.py eval --checkpoint 015000
    python run_train_eval.py eval --from-hub        # use the policy pushed to the Hub
    python run_train_eval.py eval --teleop-reset    # use leader arm to reset between episodes
    python run_train_eval.py eval --quick           # just run the policy for 30 s, no recording

Save a copy of the training log:
    python run_train_eval.py train 2>&1 | tee train_log.txt
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

import run_record as rec  # shared robot / camera / dataset config


# ---- Edit these to match your setup -----------------------------------
POLICY_NAME = "act-watch-pickplace-real"
OUTPUT_DIR = "~/watch-pickplace-sim/outputs/train/act_watch_pickplace_real"

# Training (sized for a 4 GB laptop GPU)
STEPS = 20000
BATCH_SIZE = 4          # 8 is likely to run out of memory on 4 GB. Try 2 if 4 still fails.
NUM_WORKERS = 2         # data-loader processes. System RAM is tight, so keep this low.
USE_AMP = True          # mixed precision: less GPU memory, faster on RTX cards
SAVE_FREQ = 5000        # 4 checkpoints for a 20k run
LOG_FREQ = 100          # print loss every 100 steps
CHUNK_SIZE = 50         # ~1.7 s of actions at 30 fps
N_ACTION_STEPS = 50     # must be <= CHUNK_SIZE
IMAGE_AUGMENTATION = True  # random brightness/contrast/shift; helps with few episodes
MIN_RECOMMENDED_EPISODES = 15

# Eval on the real robot
EVAL_EPISODES = 5
EVAL_EPISODE_TIME_S = 45
EVAL_RESET_TIME_S = 15
EVAL_ROOT = "~/watch-pickplace-sim/datasets/eval"
EVAL_DISPLAY = False    # live viewer (Rerun). It crashed before on WSL, so off by default.
EVAL_QUICK_DURATION_S = 60
# -------------------------------------------------------------------------


def expand(path: str) -> str:
    return os.path.expanduser(path)


def detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def print_gpu_info() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"[gpu] {props.name}, {props.total_memory / 1024**3:.1f} GB VRAM\n")
    except ImportError:
        pass


def run(cmd: list[str]) -> None:
    print("Running:\n  " + " \\\n  ".join(cmd) + "\n")

    env = os.environ.copy()
    # reduces GPU memory fragmentation -> fewer out-of-memory errors on small GPUs
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # W&B is disabled; this also stops any accidental wandb login prompt
    env["WANDB_MODE"] = "disabled"

    result = subprocess.run(cmd, env=env)  # inherit stdin/stdout (keyboard controls work)
    sys.exit(result.returncode)


def check_dataset() -> str:
    """Make sure the local dataset exists. Print episode count."""
    root = expand(rec.DATASET_ROOT)
    info_path = os.path.join(root, "meta", "info.json")
    if not os.path.exists(info_path):
        sys.exit(f"No dataset found at {root}. Record some episodes first.")

    with open(info_path) as f:
        info = json.load(f)
    n_eps = info.get("total_episodes", 0)
    n_frames = info.get("total_frames", 0)
    print(f"[dataset] {root}")
    print(f"[dataset] {n_eps} episodes, {n_frames} frames\n")

    if n_frames == 0:
        sys.exit("Dataset has 0 frames. Recording may not have finalized properly.")

    if n_eps < MIN_RECOMMENDED_EPISODES:
        print(f"WARNING: only {n_eps} episodes. {MIN_RECOMMENDED_EPISODES}+ is recommended.")
        if input("Train anyway? Type 'yes' to continue: ").strip().lower() != "yes":
            sys.exit("Aborted.")
    return root


# ---------------------------------------------------------------- train
def cmd_train(args) -> None:
    dataset_root = check_dataset()
    out_dir = expand(OUTPUT_DIR)

    if os.path.exists(out_dir):
        if not args.fresh:
            sys.exit(
                f"{out_dir} already exists.\n"
                "Use 'resume' to continue it, or 'train --fresh' to delete it and start over."
            )
        if input(f"This will DELETE {out_dir}. Type 'yes' to confirm: ").strip().lower() != "yes":
            sys.exit("Aborted.")
        shutil.rmtree(out_dir)
        print(f"Removed {out_dir}\n")

    device = detect_device()
    if device == "cpu":
        print("WARNING: no CUDA GPU found. Training on CPU will take days.")
        if input("Continue on CPU? Type 'yes': ").strip().lower() != "yes":
            sys.exit("Aborted.")
    print_gpu_info()

    hf_user = rec.get_hf_username()
    use_amp = USE_AMP and device == "cuda"

    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={hf_user}/{rec.DATASET_NAME}",
        f"--dataset.root={dataset_root}",  # train from local copy, no download
        "--dataset.video_backend=pyav",
        f"--dataset.image_transforms.enable={'true' if IMAGE_AUGMENTATION else 'false'}",
        "--policy.type=act",
        f"--output_dir={out_dir}",
        f"--job_name={POLICY_NAME}",
        f"--batch_size={BATCH_SIZE}",
        f"--num_workers={NUM_WORKERS}",
        f"--steps={STEPS}",
        "--save_checkpoint=true",
        f"--save_freq={SAVE_FREQ}",
        f"--log_freq={LOG_FREQ}",
        f"--policy.device={device}",
        f"--policy.use_amp={'true' if use_amp else 'false'}",
        f"--policy.chunk_size={CHUNK_SIZE}",
        f"--policy.n_action_steps={N_ACTION_STEPS}",
        "--wandb.enable=false",
        f"--policy.push_to_hub={'true' if args.push else 'false'}",
    ]
    if args.push:
        cmd.append(f"--policy.repo_id={hf_user}/{POLICY_NAME}")

    run(cmd)


# ---------------------------------------------------------------- resume
def cmd_resume(args) -> None:
    out_dir = expand(OUTPUT_DIR)
    cfg = os.path.join(out_dir, "checkpoints", "last", "pretrained_model", "train_config.json")
    if not os.path.exists(cfg):
        sys.exit(f"No checkpoint to resume from.\nExpected: {cfg}")

    cmd = ["lerobot-train", f"--config_path={cfg}", "--resume=true"]
    if args.steps:
        cmd.append(f"--steps={args.steps}")
    run(cmd)


# ---------------------------------------------------------------- list
def cmd_list(_args) -> None:
    ckpt_dir = os.path.join(expand(OUTPUT_DIR), "checkpoints")
    if not os.path.isdir(ckpt_dir):
        sys.exit(f"No checkpoints yet at {ckpt_dir}")

    names = sorted(n for n in os.listdir(ckpt_dir) if n != "last")
    print(f"Checkpoints in {ckpt_dir}:")
    for n in names:
        print(f"  {n}")
    last = os.path.join(ckpt_dir, "last")
    if os.path.islink(last):
        print(f"  last -> {os.readlink(last)}")


# ---------------------------------------------------------------- eval
def cmd_eval(args) -> None:
    hf_user = rec.get_hf_username()

    if args.from_hub:
        policy_path = f"{hf_user}/{POLICY_NAME}"
    else:
        policy_path = os.path.join(
            expand(OUTPUT_DIR), "checkpoints", args.checkpoint, "pretrained_model"
        )
        if not os.path.isdir(policy_path):
            sys.exit(f"Checkpoint not found: {policy_path}\nRun 'list' to see what exists.")

    # LeRobot requires eval dataset names to start with "eval_".
    # A timestamp keeps each eval run in its own folder (no "already exists" error).
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_name = f"eval_{rec.DATASET_NAME}_{stamp}"
    eval_root = os.path.join(expand(EVAL_ROOT), eval_name)

    device = args.device or detect_device()

    # Newer LeRobot: policies run through lerobot-rollout, not lerobot-record.
    cmd = [
        "lerobot-rollout",
        f"--policy.path={policy_path}",
        f"--device={device}",
        f"--robot.type={rec.ROBOT_TYPE}",
        f"--robot.port={rec.ROBOT_PORT}",
        f"--robot.id={rec.ROBOT_ID}",
        f"--robot.cameras={json.dumps(rec.CAMERAS)}",  # same cameras as training
        f"--display_data={'true' if EVAL_DISPLAY else 'false'}",
    ]

    if args.quick:
        # run the policy only, no recording
        cmd += [
            "--strategy.type=base",
            f"--task={rec.TASK_DESCRIPTION}",
            f"--duration={args.duration}",
        ]
    else:
        # episode-by-episode, recorded like lerobot-record
        cmd += [
            "--strategy.type=episodic",
            f"--dataset.repo_id={hf_user}/{eval_name}",
            f"--dataset.root={eval_root}",
            f"--dataset.num_episodes={args.episodes}",
            f"--dataset.episode_time_s={EVAL_EPISODE_TIME_S}",
            f"--dataset.reset_time_s={EVAL_RESET_TIME_S}",
            f"--dataset.single_task={rec.TASK_DESCRIPTION}",
            "--dataset.push_to_hub=false",
        ]

    if args.teleop_reset and not args.quick:
        # leader arm only drives the follower during the reset phase
        cmd += [
            f"--teleop.type={rec.TELEOP_TYPE}",
            f"--teleop.port={rec.TELEOP_PORT}",
            f"--teleop.id={rec.TELEOP_ID}",
        ]

    print(f"[eval] policy : {policy_path}")
    print(f"[eval] device : {device}")
    if args.quick:
        print(f"[eval] mode   : quick (no recording), {args.duration} s\n")
    else:
        print(f"[eval] saving : {eval_root}\n")
        print("Keys: RIGHT arrow = end episode early, LEFT arrow = re-record, ESC = stop.")
    print("Hands off the leader arm while the policy runs. Keep a hand near the power switch.\n")
    run(cmd)


def main():
    parser = argparse.ArgumentParser(description="ACT training + real-robot eval for SO-101")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train ACT on the local dataset")
    p_train.add_argument("--push", action="store_true", help="push the policy to the Hub")
    p_train.add_argument("--fresh", action="store_true", help="delete old output dir first")
    p_train.set_defaults(func=cmd_train)

    p_resume = sub.add_parser("resume", help="resume from the last checkpoint")
    p_resume.add_argument("--steps", type=int, help="new TOTAL step count (e.g. 40000)")
    p_resume.set_defaults(func=cmd_resume)

    p_list = sub.add_parser("list", help="list saved checkpoints")
    p_list.set_defaults(func=cmd_list)

    p_eval = sub.add_parser("eval", help="run the trained policy on the real arm")
    p_eval.add_argument("--checkpoint", default="last", help="e.g. 015000 (default: last)")
    p_eval.add_argument("--from-hub", action="store_true", help="load policy from the Hub")
    p_eval.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    p_eval.add_argument("--device", choices=["cuda", "cpu"], help="default: auto-detect")
    p_eval.add_argument("--teleop-reset", action="store_true",
                        help="connect leader arm to reset the scene between episodes")
    p_eval.add_argument("--quick", action="store_true",
                        help="just run the policy, no recording (lerobot-rollout base strategy)")
    p_eval.add_argument("--duration", type=int, default=EVAL_QUICK_DURATION_S,
                        help="seconds to run in --quick mode (0 = until ESC/Ctrl+C)")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()