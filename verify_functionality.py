#!/usr/bin/env python3
"""
Run verification steps from README "## Verify that everything works".
Runs CUDA check then training for all configs in configs/example_configs/ that start with "test".
Output is written to verification.log (or path given as first argument).
Exit code 0 on success, non-zero on failure.
"""
import os
import subprocess
import sys
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "verification.log"
if len(sys.argv) > 1:
    LOG_PATH = Path(sys.argv[1])

# Timeout per training config (seconds)
TIMEOUT_PER_CONFIG = 600

def run(cmd, cwd=None, env=None, timeout=TIMEOUT_PER_CONFIG):
    """Run command, return (stdout+stderr, returncode)."""
    p = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd or Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env if env is not None else os.environ,
    )
    out = (p.stdout or "") + (p.stderr or "")
    return out, p.returncode

def main():
    lines = []
    repo_root = Path(__file__).resolve().parent
    configs_dir = repo_root / "configs" / "example_configs"

    # Clear log so this run is the only content (check_logs reads this file)
    with open(LOG_PATH, "w"):
        pass

    env_geotiff = {**os.environ, "GTIFF_SRS_SOURCE": "EPSG"}

    # 1. CUDA check
    lines.append("=== CUDA check ===")
    out, ret = run(
        f'{sys.executable} -c "import torch; print(\'CUDA available:\', torch.cuda.is_available()); print(\'Device:\', torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'N/A\')"',
        cwd=repo_root,
        timeout=30,
    )
    lines.append(out)
    lines.append(f"Exit code: {ret}\n")
    if ret != 0:
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(lines))
        return ret

    # 2. Find all configs whose filename starts with "test"
    test_configs = sorted(configs_dir.glob("test*.ini"))
    if not test_configs:
        lines.append("=== No test* configs found ===\n")
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(lines))
        return 0

    for config_path in test_configs:
        rel = config_path.relative_to(repo_root)
        lines.append(f"=== train.py --config {rel} ===")
        out, ret = run(
            f"{sys.executable} src/ML_sdfi_fastai2/train.py --config {rel}",
            cwd=repo_root,
            env=env_geotiff,
        )
        lines.append(out)
        lines.append(f"Exit code: {ret}\n")
        if ret != 0:
            with open(LOG_PATH, "w") as f:
                f.write("\n".join(lines))
            return ret

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(lines))
    return 0

if __name__ == "__main__":
    sys.exit(main())
