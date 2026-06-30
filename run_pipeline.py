#!/usr/bin/env python3
"""One-command reproduction of the full segmentation study.

Each experiment carries forward the best configuration from the previous stage
and communicates with the next through files written under ``results/`` (the
immutable data split, ``best_encoder.json``, ``best_loss.json``, ...). Running
this script reproduces the paper end to end:

    1   data discovery + placenta-level split        (results/experiment_1_data_split/)
    2a  encoder comparison      (14 encoders)         -> best_encoder.json
    2b  decoder comparison      (5 decoders)          -> best_decoder.json
    3   loss-function comparison(9 losses)            -> best_loss.json
    4   augmentation comparison (7 strategies)        -> best_augmentation.json
    5   data-efficiency curve   (1-13 placentas)
    6   downstream morphometric validation (held-out test set)
    6v2 morphometrics recomputed at native resolution

Usage
-----
    python run_pipeline.py                 # run the whole pipeline, in order
    python run_pipeline.py --list          # list the stages and exit
    python run_pipeline.py --only 1        # run a single stage
    python run_pipeline.py --from 3        # resume from stage 3 to the end
    python run_pipeline.py --from 2a --to 4
    python run_pipeline.py --dry-run       # print the commands without running

Prerequisites
-------------
* ``data/cd31_images/`` and ``data/masks/`` populated (see data/README.md).
* Dependencies installed:  pip install -r requirements.txt
* A CUDA GPU is strongly recommended for stages 2a-6 (training stages).

Stage 1 needs only the data and runs in seconds; the training stages
(2a-6) take hours on a single GPU. Stage 6v2 reuses stage 6's predictions.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(HERE, "experiments")

# Ordered pipeline. Each tuple: (stage key, script filename, one-line description).
STAGES = [
    ("1",   "run_experiment_1_split.py",        "Data discovery + placenta-level split (seed=42)"),
    ("2a",  "run_experiment_2a_encoder.py",     "Encoder comparison (14 encoders x 3 seeds)"),
    ("2b",  "run_experiment_2b_decoder.py",     "Decoder comparison (5 decoders x 3 seeds)"),
    ("3",   "run_experiment_3_loss.py",         "Loss-function comparison (9 losses x 3 seeds)"),
    ("4",   "run_experiment_4_augmentation.py", "Augmentation comparison (7 strategies)"),
    ("5",   "run_experiment_5_efficiency.py",   "Data-efficiency curve (1-13 placentas)"),
    ("6",   "run_experiment_6_validation.py",   "Downstream morphometric validation (test set)"),
    ("6v2", "run_experiment_6_validation_v2.py","Morphometrics at native resolution"),
]
STAGE_KEYS = [k for k, _, _ in STAGES]


def _index(key):
    if key not in STAGE_KEYS:
        sys.exit(f"Unknown stage '{key}'. Valid stages: {', '.join(STAGE_KEYS)}")
    return STAGE_KEYS.index(key)


def select_stages(args):
    if args.only:
        return [STAGES[_index(args.only)]]
    start = _index(args.from_stage) if args.from_stage else 0
    end = _index(args.to_stage) if args.to_stage else len(STAGES) - 1
    if start > end:
        sys.exit(f"--from ({args.from_stage}) comes after --to ({args.to_stage}).")
    return STAGES[start:end + 1]


def main():
    p = argparse.ArgumentParser(
        description="Run the placental blood-space segmentation pipeline end to end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--only", metavar="STAGE", help="Run exactly one stage (e.g. 2a).")
    p.add_argument("--from", dest="from_stage", metavar="STAGE", help="First stage to run.")
    p.add_argument("--to", dest="to_stage", metavar="STAGE", help="Last stage to run.")
    p.add_argument("--list", action="store_true", help="List stages and exit.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = p.parse_args()

    if args.list:
        print("Pipeline stages (run in this order):\n")
        for key, script, desc in STAGES:
            print(f"  [{key:>3}]  {desc}")
            print(f"         {os.path.join('experiments', script)}")
        return

    stages = select_stages(args)
    print(f"Will run {len(stages)} stage(s): {', '.join(k for k, _, _ in stages)}\n")

    for key, script, desc in stages:
        script_path = os.path.join(EXP_DIR, script)
        if not os.path.exists(script_path):
            sys.exit(f"Missing script: {script_path}")
        print("=" * 78)
        print(f"STAGE {key}: {desc}")
        print(f"  $ python {os.path.relpath(script_path, HERE)}")
        print("=" * 78, flush=True)

        if args.dry_run:
            continue

        t0 = time.time()
        # Run from the repo root so the scripts' BASE_DIR-relative paths resolve.
        result = subprocess.run([sys.executable, script_path], cwd=HERE)
        elapsed = (time.time() - t0) / 60.0
        if result.returncode != 0:
            sys.exit(f"\nStage {key} failed (exit {result.returncode}) after {elapsed:.1f} min. "
                     f"Pipeline halted.")
        print(f"\nStage {key} completed in {elapsed:.1f} min.\n", flush=True)

    print("Done." + ("" if args.dry_run else " All requested stages completed successfully."))


if __name__ == "__main__":
    main()
