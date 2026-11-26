"""Writes evaluation/5.1a/output/manifest.txt: provenance of the 5.1a run (git state, versions, checks)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

OUT = Path(__file__).resolve().parent / "output"


def git(*a):
    return subprocess.run(["git", *a], cwd=c.REPO, capture_output=True, text=True).stdout.strip()


def main():
    info = json.loads((OUT / "run_info.json").read_text())
    dirty = git("status", "--porcelain", "--", "evaluation/5.1a", "metrics.py", "occlusion_handling.py", "tests/test_5_1a.py")
    dirty = "\n".join(l for l in dirty.splitlines() if "/output/" not in l)
    lines = ["Step 5.1a run manifest (evaluation/5.1a; PROTOCOL.md was committed before the sweep, 78cacd7)",
             "NOTHING here is validated: speed-profile stability check on fixed windows, not movement segments; no ground truth; five clips.",
             f"git HEAD: {git('rev-parse', 'HEAD')}",
             f"code/inputs modified since HEAD (outside output/): {'yes' if dirty else 'no'}"]
    if dirty:
        lines.append(dirty)
    lines += [f"numpy {np.__version__}, scipy {scipy.__version__}",
              f"metrics.py sha256 {hashlib.sha256((c.REPO / 'metrics.py').read_bytes()).hexdigest()}",
              f"occlusion_handling.py sha256 {hashlib.sha256((c.REPO / 'occlusion_handling.py').read_bytes()).hexdigest()}",
              f"raw_results.npz sha256 {hashlib.sha256((OUT / 'raw_results.npz').read_bytes()).hexdigest()} (gitignored, regenerable)",
              "", "== run", json.dumps(info), "",
              "== checks"]
    for name in c.CLIPS:
        clip = c.Clip(name)
        n, bad = c.check_filled_equal(clip)
        lines.append(f"{name}: re-run fill vs evaluation/3.2/output_full_lock1_d030/{name}_filled.csv: {n} rows compared, {bad} different")
    lines.append(f"speed arrays vs metrics.speed_profile (noise-free, unfiltered, E-dL): {info['speed_profile_check_windows']} windows, "
                 f"{info['speed_profile_check_mismatch']} mismatches")
    (OUT / "manifest.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
