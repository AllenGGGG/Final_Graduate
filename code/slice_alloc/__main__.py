"""The single supported entry for the first-part native allocation study."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="低空多模态网络切片初始分配")
    p.add_argument("command", nargs="?", default="verify",
                   choices=("all", "train", "evaluate", "verify"))
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    package = Path(__file__).resolve().parent
    output = Path(os.environ.get("LOWALT_ALLOCATION_OUTPUT", str(package / "results/final"))).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LOWALT_ALLOCATION_OUTPUT=str(output))
    def run(module, *extra):
        command=[sys.executable, "-m", module, *extra, "--workers", str(args.workers)]
        subprocess.run(command,
                       cwd=package.parent, env=env, check=True)
    if args.command in ("train", "all") and not (output / "training_complete.json").exists():
        run("slice_alloc.tools.allocation_confirm_train")
    if args.command == "train":
        return
    if args.command in ("evaluate", "all") and not (output / "evaluation.json").exists():
        run("slice_alloc.tools.allocation_confirm_evaluate", "evaluate")
    if args.command in ("verify", "all", "evaluate"):
        run("slice_alloc.tools.allocation_confirm_evaluate", "verify")


if __name__ == "__main__":
    main()
