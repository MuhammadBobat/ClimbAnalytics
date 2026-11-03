"""
Run manifest for a gated-lock pipeline run (default validation/output_full_lock1/): git HEAD and whether the code
was clean, the rule parameters, inputs and library versions. Written next to the run (gitignored with it) and
copied to evaluation/lock_rule/output_lock1/run_manifest.txt (committed).

Usage:
    python3 evaluation/lock_rule/run_manifest.py [--run-dir validation/output_full_lock1]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from validation.analyze_occlusion_gaps import MANIFEST_LIBRARIES, _git, _library_version, sha256_file  # noqa: E402

CODE = ["person_lock.py", "validation/validate_vitpose.py"]   # the code that determines the run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("validation/output_full_lock1"))
    args = ap.parse_args()
    dirty = _git("status", "--porcelain", "--", *CODE)
    snaps = sorted((Path.home() / ".cache/huggingface/hub/models--usyd-community--vitpose-plus-large/snapshots").glob("*"))
    summary = (args.run_dir / "summary.csv").read_text().rstrip("\n").split("\n")
    L = [
        "Gated-lock pipeline run manifest (evaluation/lock_rule/run_manifest.py)",
        f"date written: {date.today().isoformat()}",
        f"run folder: {args.run_dir.as_posix()} (new folder; validation/output and validation/output_full not written)",
        "command: python3 validation/validate_vitpose.py --lock-rule gated --output-dir " + args.run_dir.as_posix(),
        "", "== Rule (DECISIONS.md D-027, status proposed)",
        (args.run_dir / "run_info.txt").read_text().rstrip("\n"),
        "selection order applied after seeing the stage-2 results (not pre-registered); see evaluation/lock_rule/output_full/manifest.txt",
        "", "== Code state",
        f"git HEAD at run start and now: {_git('rev-parse', 'HEAD')}",
        "uncommitted changes to person_lock.py and validation/validate_vitpose.py (the code that determines the run) when written: "
        + (dirty.replace("\n", "; ") or "none (clean tree)"),
        "before launch (recorded then): git status of person_lock.py, validate_vitpose.py and evaluation/lock_rule showed no changes, HEAD dedb76d",
        "analysis scripts written after the run (compare_gated_run.py, run_manifest.py) do not affect it",
        f"person_lock.py sha256 {sha256_file(Path('person_lock.py'))}",
        f"validation/validate_vitpose.py sha256 {sha256_file(Path('validation/validate_vitpose.py'))}",
        "", "== Inputs",
        f"person detector: validation/weights/yolov8n.pt sha256 {sha256_file(Path('validation/weights/yolov8n.pt'))}",
        "pose model: usyd-community/vitpose-plus-large, dataset_index 0; local HF snapshot(s): "
        + (", ".join(p.name for p in snaps) or "not found"),
        "clip hashes: see validation/box_cache/manifest.txt", "", "== Summary (summary.csv)", *summary,
        "", "== Environment (versions at manifest time; the device used by the run is auto-detected as in validate_vitpose.py "
        "and is not recorded by it)",
        *(f"{lib}: {_library_version(*(('opencv-python', 'opencv-python-headless') if lib == 'opencv-python' else (lib,)))}"
          for lib in MANIFEST_LIBRARIES), ""]
    text = "\n".join(L)
    (args.run_dir / "manifest.txt").write_text(text)
    out = Path("evaluation/lock_rule/output_lock1")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_manifest.txt").write_text(text)
    print(f"written {args.run_dir / 'manifest.txt'} and {out / 'run_manifest.txt'}")


if __name__ == "__main__":
    main()
