"""
Per-frame detail table for one clip and frame range: proxy CoM and segment-model CoM, both speeds
(step t-1 -> t), and the raw hip and shoulder keypoint positions and confidences (left and
right). Reuses compare_com_models.py (same exclusions) and make_overlay_clips.load_raw. Numbers
only, no interpretation. Raw, unsmoothed, no gap-filling; an empty cell means undefined.

Usage:
    python3 evaluation/2.2b/frame_detail_table.py --clip IMG_2420 --start 68 --end 84
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

import compare_com_models as cm  # noqa: E402
import make_overlay_clips as oc  # noqa: E402
from metrics import SEX_TABLES, compute_velocity  # noqa: E402
from validation.analyze_occlusion_gaps import _git, load_exclusions, sha256_file  # noqa: E402

JOINTS = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]


def build_rows(c, raw, sex: str, start: int, end: int) -> tuple[list[str], list[list]]:
    vp, vs = compute_velocity(c.proxy), compute_velocity(c.seg[sex])
    header = ["frame", "proxy_x", "proxy_y", f"seg_{sex}_x", f"seg_{sex}_y",
              "speed_proxy_px_s", "speed_segment_px_s"]
    for j in JOINTS:
        header += [f"{j}_x", f"{j}_y", f"{j}_conf"]
    rows = []
    for t in range(start, end + 1):
        p, s = c.proxy.positions[t], c.seg[sex].positions[t]
        row = [t, cm._f(p[0]) if p else "", cm._f(p[1]) if p else "",
               cm._f(s[0]) if s else "", cm._f(s[1]) if s else "",
               cm._f(vp[t]), cm._f(vs[t])]
        for j in JOINTS:
            x, y, conf = raw.get(t, {}).get(j, (None, None, None))
            row += [cm._f(x), cm._f(y), "" if conf is None else f"{conf:.4f}"]
        rows.append(row)
    return header, rows


def build_manifest(csv_path: Path, exclusions_path: Path, clip: str, start: int, end: int, sex: str,
                   today: str | None = None) -> str:
    """Run provenance: input keypoint CSV (SHA-256), exclusion file, code state at run time."""
    from datetime import date
    files = ["metrics.py", "evaluation/2.2b/frame_detail_table.py", "evaluation/2.2b/compare_com_models.py",
             "evaluation/2.2b/make_overlay_clips.py", "validation/frame_exclusions.csv"]
    dirty = _git("status", "--porcelain", "--", *files)
    L = ["Frame detail table manifest (evaluation/2.2b/frame_detail_table.py)",
         f"date: {today or date.today().isoformat()}",
         f"clip: {clip}, frames {start}-{end}, segment-model sex table: {sex}",
         f"input keypoint CSV: {csv_path.as_posix()} sha256 "
         + (sha256_file(csv_path) if csv_path.exists() else "not found"),
         "columns: proxy and segment-model CoM (x, y), both speeds (step t-1 -> t), raw hip and shoulder "
         "keypoint positions and confidences; an empty cell means undefined. Raw, unsmoothed, no gap-filling.",
         "", "== Frame exclusion file"]
    if exclusions_path.exists():
        L += [f"path: {exclusions_path.as_posix()}", "contents (verbatim):",
              exclusions_path.read_text().rstrip("\n")]
    else:
        L.append("none")
    L += ["", "== Code (at run time)", f"git HEAD: {_git('rev-parse', 'HEAD')}",
          "files modified since HEAD (metrics.py, this script, compare_com_models.py, make_overlay_clips.py, "
          "frame_exclusions.csv): " + ("unknown" if dirty == "unknown" else (dirty.replace("\n", "; ") if dirty else "none")),
          "this script's git blob hash: " + _git("hash-object", "evaluation/2.2b/frame_detail_table.py")]
    return "\n".join(L) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", required=True, help="e.g. IMG_2420")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("validation/output_full"))
    parser.add_argument("--eval-dir", type=Path, default=Path("evaluation/2.2b"))
    parser.add_argument("--exclusions", type=Path, default=Path("validation/frame_exclusions.csv"))
    parser.add_argument("--sex", choices=SEX_TABLES, default="male")
    args = parser.parse_args()

    csv_path = args.input_dir / f"{args.clip}_keypoints.csv"
    c = cm.compare_clip(csv_path, load_exclusions(args.exclusions).get(f"{args.clip}.MOV"))
    header, rows = build_rows(c, oc.load_raw(csv_path), args.sex, args.start, args.end)
    out = args.eval_dir / args.input_dir.resolve().name / f"{args.clip}_frames_{args.start}-{args.end}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    manifest = out.with_suffix(".manifest.txt")
    manifest.write_text(build_manifest(csv_path, args.exclusions, args.clip, args.start, args.end, args.sex))
    print(f"Written to {out} and {manifest}")


if __name__ == "__main__":
    main()
