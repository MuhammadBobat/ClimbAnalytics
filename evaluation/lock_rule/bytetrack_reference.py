"""
One UNTUNED ByteTrack reference run (plan: ByteTrack as a reference next to the rules, not part of the sweep;
DECISIONS.md D-024 option 5, D-027). Ultralytics' tracker with library defaults on the same YOLOv8n person
detector: model.track(..., classes=[0], tracker="bytetrack.yaml") with the library's default confidence and
tracker settings; nothing is set here except what is needed to pick the climber:

  Lock rule used on top of the track IDs (the minimum needed; stated, not tuned): the climber is the track ID of
  the highest-confidence tracked box on the first frame that has one; the chosen box is that ID's box on each
  frame, none if the ID is absent; after TRACK_BUFFER_DEFAULT = 30 consecutive frames without the ID (the library's
  default track_buffer, read from the tracker config at run time and recorded) the climber is re-acquired as
  the highest-confidence tracked box, as on the first frame.

Output (evaluation/lock_rule/output_full/): bytetrack_boxes_<clip>.csv (frame, x, y, w, h, conf, track_id),
bytetrack_summary.csv (same per-clip metrics as the sweep, against the legacy choice and the provisional 2419
reference track), bytetrack_manifest.txt.

Usage:
    python3 evaluation/lock_rule/bytetrack_reference.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lock_common import (  # noqa: E402
    CLIPS, LockParams, height_ratio_flags, iou, jump_stats, load_cache, reference_track_2419, run_rule,
    runs_of, same_box)
from person_lock import Box  # noqa: E402
from validation.analyze_occlusion_gaps import _git, _library_version, sha256_file  # noqa: E402

OUT = Path("evaluation/lock_rule/output_full")


def climber_track(per_frame: list[list[tuple[int | None, Box]]], buffer: int) -> list[tuple[Box | None, int | None]]:
    """Follow one track ID (see the module docstring)."""
    cid, absent, out = None, 0, []
    for tracked in per_frame:
        ids = [(i, b) for i, b in tracked if i is not None]
        if cid is None or absent >= buffer:
            if ids:
                cid, b = max(ids, key=lambda ib: ib[1].conf)[0], max(ids, key=lambda ib: ib[1].conf)[1]
                absent = 0
                out.append((b, cid))
                continue
        hit = next((b for i, b in ids if i == cid), None)
        if hit is not None:
            absent = 0
            out.append((hit, cid))
        else:
            absent += 1 if cid is not None else 0
            out.append((None, None))
    return out


def main() -> None:
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.utils import ROOT

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tracker_cfg = yaml.safe_load((ROOT / "cfg" / "trackers" / "bytetrack.yaml").read_text())
    buffer = int(tracker_cfg["track_buffer"])
    detector = YOLO("validation/weights/yolov8n.pt")
    frames = {c: load_cache(c) for c in CLIPS}
    legacy = {c: run_rule(frames[c], LockParams.legacy())[0] for c in CLIPS}
    ref = reference_track_2419(frames["IMG_2419"], legacy["IMG_2419"])
    rows = []
    for clip in CLIPS:
        per_frame = []
        for r in detector.track(source=f"footage/test/{clip}.MOV", stream=True, classes=[0],
                                tracker="bytetrack.yaml", device=device, verbose=False):
            tracked = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [None] * len(conf)
                for (x1, y1, x2, y2), c, i in zip(xyxy, conf, ids):
                    tracked.append((None if i is None else int(i), Box(float(x1), float(y1), float(x2 - x1),
                                                                      float(y2 - y1), float(c))))
            per_frame.append(tracked)
        res = climber_track(per_frame, buffer)
        chosen = [b for b, _ in res]
        with open(OUT / f"bytetrack_boxes_{clip}.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["frame", "x", "y", "w", "h", "conf", "track_id"])
            for t, (b, i) in enumerate(res):
                w.writerow([t] + ([f"{b.x:.1f}", f"{b.y:.1f}", f"{b.w:.1f}", f"{b.h:.1f}", f"{b.conf:.3f}", i]
                                  if b else ["", "", "", "", "", ""]))
        leg = legacy[clip]
        drops = runs_of([b is None for b in chosen])
        gained = sum(a is not None and b is None for a, b in zip(chosen, leg))
        lost = sum(a is None and b is not None for a, b in zip(chosen, leg))
        replaced = sum(a is not None and b is not None and not (iou(a, b) >= 0.9) for a, b in zip(chosen, leg))
        j = jump_stats(chosen)
        row = {"clip": clip, "n_frames": len(chosen), "n_dropped": sum(b is None for b in chosen),
               "longest_drop": max((e - s + 1 for s, e in drops), default=0), "gained_vs_legacy": gained,
               "lost_vs_legacy": lost, "box_differs_from_legacy_iou_lt_0_9": replaced,
               "jump_max": f"{j.get('max', float('nan')):.4f}", "jump_p99": f"{j.get('p99', float('nan')):.4f}",
               "n_jump_over_0_3": j.get("n_over_0_3", 0), "n_hratio_flags": len(height_ratio_flags(chosen)),
               "win2419_correct": "", "win2419_wrong": "", "win2419_dropped": ""}
        if clip == "IMG_2419":
            c = w_ = d = 0
            for t in range(592, 615):
                b, (r_, _) = chosen[t], ref[t]
                if b is None:
                    d += 1
                elif r_ is not None and iou(b, r_) >= 0.5:
                    c += 1
                else:
                    w_ += 1
            row.update(win2419_correct=c, win2419_wrong=w_, win2419_dropped=d)
        rows.append(row)
        print(row, flush=True)
    with open(OUT / "bytetrack_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    dirty = _git("status", "--porcelain", "--", "evaluation/lock_rule/bytetrack_reference.py")
    (OUT / "bytetrack_manifest.txt").write_text("\n".join([
        "ByteTrack reference run manifest (evaluation/lock_rule/bytetrack_reference.py)",
        "untuned: ultralytics model.track(stream=True, classes=[0], tracker='bytetrack.yaml'), library defaults",
        f"tracker config read at run time: {tracker_cfg}", f"track_buffer used for re-acquisition: {buffer}",
        f"device: {device}", f"weights: validation/weights/yolov8n.pt sha256 {sha256_file(Path('validation/weights/yolov8n.pt'))}",
        "climber = ID of the highest-confidence tracked box on the first frame with one; re-acquired the same way "
        "after track_buffer consecutive frames without that ID",
        "reference track for the 2419 window is PROVISIONAL (reference_track_2419.csv)",
        f"git HEAD: {_git('rev-parse', 'HEAD')}",
        "script modified since HEAD: " + ("unknown" if dirty == "unknown" else ("yes" if dirty else "no")),
        f"ultralytics: {_library_version('ultralytics')}", f"torch: {_library_version('torch')}", ""]))


if __name__ == "__main__":
    main()
