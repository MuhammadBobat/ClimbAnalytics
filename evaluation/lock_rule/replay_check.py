"""
Replay-equals-baseline check (plan step 1; DECISIONS.md D-024). Runs the LEGACY rule on the cached boxes
(validation/box_cache/) and compares, for every frame, with validation/output_full/*_keypoints.csv:
  - frames with a chosen box: the bbox columns as written (one decimal) and n_person_boxes (= cached
    boxes with conf >= 0.5);
  - frames with no row there: the legacy replay must also choose no box.
Exit status 1 on any difference. Writes evaluation/lock_rule/output_full/replay_check.txt.

Usage:
    python3 evaluation/lock_rule/replay_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lock_common import CLIPS, LockParams, fmt1, load_baseline, load_cache, run_rule  # noqa: E402


def check_clip(clip: str) -> dict:
    frames = load_cache(clip)
    base = load_baseline(clip)
    chosen, _ = run_rule(frames, LockParams.legacy())
    bad_box = bad_n = bad_missing = bad_extra = 0
    first_bad = None
    for t, box in enumerate(chosen):
        n_main = sum(1 for b in frames[t] if b.conf >= 0.5)
        if t in base:
            want_box, want_n = base[t]
            if box is None or fmt1(box) != want_box:
                bad_box += 1
                first_bad = first_bad if first_bad is not None else t
            if n_main != want_n:
                bad_n += 1
                first_bad = first_bad if first_bad is not None else t
        elif box is not None:
            bad_extra += 1
            first_bad = first_bad if first_bad is not None else t
    bad_missing = sum(1 for t in base if t >= len(frames))
    return {"clip": clip, "cache_frames": len(frames), "baseline_frames_with_box": len(base),
            "legacy_frames_with_box": sum(b is not None for b in chosen), "box_mismatch": bad_box,
            "n_person_boxes_mismatch": bad_n, "extra_box_where_baseline_has_none": bad_extra,
            "baseline_frames_beyond_cache": bad_missing, "first_bad_frame": first_bad}


def main() -> None:
    out_dir = Path("evaluation/lock_rule/output_full")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [check_clip(c) for c in CLIPS]
    ok = all(r["box_mismatch"] == r["n_person_boxes_mismatch"] == r["extra_box_where_baseline_has_none"]
             == r["baseline_frames_beyond_cache"] == 0 for r in rows)
    lines = ["Replay-equals-baseline check (evaluation/lock_rule/replay_check.py)",
             "legacy rule on validation/box_cache/ vs validation/output_full/*_keypoints.csv",
             f"result: {'PASS' if ok else 'FAIL'}", ""]
    for r in rows:
        lines.append(", ".join(f"{k}={v}" for k, v in r.items()))
    (out_dir / "replay_check.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
