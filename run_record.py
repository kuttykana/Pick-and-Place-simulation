#!/usr/bin/env python3
"""
Usage:
    python run_record.py                # record a new batch, push to Hub
    python run_record.py --resume       # resume the same dataset
    python run_record.py --no-push      # keep this batch local only
"""

import argparse
import json
import subprocess
import sys


# Edit these to match your setup 
ROBOT_TYPE = "so101_follower"
ROBOT_PORT = "/dev/ttyACM1"
ROBOT_ID = "my_awesome_follower_arm"
TELEOP_TYPE = "so101_leader"
TELEOP_PORT = "/dev/ttyACM0"
TELEOP_ID = "my_awesome_leader_arm"

CAMERAS = {
    # NOTE: use the conventional naming of "front" and "overhead" for the two cameras, so that the dataset can be used with the default camera selection in lerobot-dataset.
    "front": {  # wrist / gripper camera
        "type": "opencv",
        "index_or_path": "/dev/video4",
        "width": 640,
        "height": 480,
        "fps": 30,
        "fourcc": "MJPG",
        "backend": "V4L2",
        "warmup_s": 3,
    },
    "overhead": {  # top-down camera
        "type": "opencv",
        "index_or_path": "/dev/video0",
        "width": 640,
        "height": 480,
        "fps": 30,
        "fourcc": "MJPG",
        "backend": "V4L2",
        "warmup_s": 3,
    },
}

DATASET_NAME = "watch-pickplace-real"
DATASET_ROOT = "~/watch-pickplace-sim/datasets/watch-pickplace-real"
TASK_DESCRIPTION = "Pick up the watch and place it in the tray"
EPISODES_PER_BATCH = 15

# rerun's GPU viewer crashed on this WSL setup.
# So set to True + DISPLAY_MODE="foxglove" to try the
# browser-based viewer instead, which usually works better under WSL.
DISPLAY_DATA = True
DISPLAY_MODE = "foxglove"


def get_hf_username() -> str:
    """Fetch the logged-in HF username. Fails if not logged in."""
    import os
    import re

    env = os.environ.copy()
    env["NO_COLOR"] = "1"

    try:
        result = subprocess.run(
            ["hf", "auth", "whoami"], capture_output=True, text=True, check=True, env=env
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit(
            "Not logged in to Hugging Face (or 'hf' CLI not found).\n"
            "Run: hf auth login --token <your_token> --add-to-git-credential"
        )
    # strip any leftover ANSI escape codes just in case NO_COLOR is ignored
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    clean_output = ansi_escape.sub("", result.stdout)

    # output looks like:
    #   ✓ Logged in
    #     user: Neelesh123
    username = None
    for line in clean_output.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("user:"):
            username = line.split(":", 1)[-1].strip()
            break
    if not username:
        sys.exit(f"Could not parse HF username from 'hf auth whoami' output:\n{clean_output}")
    return username


def dataset_already_exists(dataset_root: str) -> bool:
    import os

    return os.path.exists(os.path.join(dataset_root, "meta", "info.json"))


def build_command(hf_user: str, resume: bool, push: bool) -> list[str]:
    import os

    repo_id = f"{hf_user}/{DATASET_NAME}"
    cameras_json = json.dumps(CAMERAS)
    dataset_root = os.path.expanduser(DATASET_ROOT)

    # auto-resume if the dataset folder already has a real LeRobotDataset in it,
    # so you don't have to remember --resume across sessions
    already_exists = dataset_already_exists(dataset_root)
    should_resume = resume or already_exists
    if already_exists and not resume:
        print(f"[auto-resume] Existing dataset found at {dataset_root} — resuming.\n")

    cmd = [
        "lerobot-record",
        f"--robot.type={ROBOT_TYPE}",
        f"--robot.port={ROBOT_PORT}",
        f"--robot.id={ROBOT_ID}",
        f"--robot.cameras={cameras_json}",
        f"--teleop.type={TELEOP_TYPE}",
        f"--teleop.port={TELEOP_PORT}",
        f"--teleop.id={TELEOP_ID}",
        f"--display_data={'true' if DISPLAY_DATA else 'false'}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        f"--dataset.num_episodes={EPISODES_PER_BATCH}",
        f"--dataset.single_task={TASK_DESCRIPTION}",
        f"--dataset.push_to_hub={'true' if push else 'false'}",
    ]
    if DISPLAY_DATA:
        cmd.append(f"--display_mode={DISPLAY_MODE}")
    if should_resume:
        cmd.append("--resume=true")
    return cmd


def main():
    import os
    import shutil

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="resume the same dataset (auto-detected normally)")
    parser.add_argument("--no-push", action="store_true", help="skip pushing this batch to the Hub")
    parser.add_argument("--fresh", action="store_true", help="wipe the existing dataset folder and start over")
    args = parser.parse_args()

    if args.fresh:
        dataset_root = os.path.expanduser(DATASET_ROOT)
        if os.path.exists(dataset_root):
            confirm = input(f"This will DELETE {dataset_root}. Type 'yes' to confirm: ")
            if confirm.strip().lower() != "yes":
                sys.exit("Aborted.")
            shutil.rmtree(dataset_root)
            print(f"Removed {dataset_root}\n")

    hf_user = get_hf_username()
    cmd = build_command(hf_user, resume=args.resume, push=not args.no_push)

    print("Running:\n  " + " \\\n  ".join(cmd) + "\n")

    # inherit stdin/stdout so recording keyboard controls (n / r / q) work
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()