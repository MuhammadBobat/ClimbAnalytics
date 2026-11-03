"""
Manifest for the lock-rule evaluation (DECISIONS.md D-024, D-027): what was run, with which grids, filters and
selection order, on which inputs, with the SHA-256 of every script, input and output file so the record does not
depend on whether the working tree was clean when it was written.

Writes evaluation/lock_rule/output_full/manifest.txt.

Usage:
    python3 evaluation/lock_rule/lock_manifest.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import simulate_gaps as sg  # noqa: E402
import sweep_lock_rule as sw  # noqa: E402
from validation.analyze_occlusion_gaps import MANIFEST_LIBRARIES, _git, _library_version, sha256_file  # noqa: E402

OUT = Path("evaluation/lock_rule/output_full")


def main() -> None:
    scripts = ["person_lock.py", "validation/validate_vitpose.py", *sorted(f"evaluation/lock_rule/{p.name}"
                                                                          for p in HERE.glob("*.py"))]
    inputs = sorted(Path("validation/box_cache").glob("*"))
    outputs = sorted(p for p in OUT.glob("*") if p.name != "manifest.txt" and p.is_file())
    L = [
        "Lock-rule evaluation manifest (evaluation/lock_rule/lock_manifest.py)",
        f"date: {date.today().isoformat()}",
        "decisions: DECISIONS.md D-024 (failure), D-027 (proposed rule); nothing here is adopted",
        "", "== What was run",
        "1. cache_boxes.py: YOLOv8n person boxes at conf >= 0.05 for every frame of the five clips (validation/box_cache/).",
        "2. replay_check.py: the legacy rule on the cache reproduced validation/output_full (box columns and n_person_boxes) "
        "on all 5,886 frames, 0 mismatches (replay_check.txt).",
        "3. sweep_lock_rule.py: full factorial of the rule family on the cache; hard filters (i) IMG_2419 592-614 has no wrong "
        "frame against the author-CONFIRMED reference track (reference_track_2419.csv), (ii) IMG_2417 identical to legacy, "
        "(iii) first 100 frames unchanged on every frame the legacy rule tracked (the literal 'identical' count is also reported).",
        "4. simulate_gaps.py: climber hidden in place in real cached boxes for L in " + ", ".join(map(str, sg.LENGTHS))
        + f" frames, trial starts every {sg.STEP} frames from {sg.FIRST}, margin {sg.MARGIN} frames, hide IoU {sg.HIDE_IOU}, "
        f"re-acquisition IoU {sg.REACQ_IOU}. LIMITATION: the climber never reappears elsewhere; the clips contain no real gap "
        "longer than 8 frames.",
        "5. select_setting.py: the author's selection order (stated 2026-11-03, AFTER the stage-2 results were seen; NOT "
        "pre-registered): a) hard filters; b) zero accepted non-climber boxes at every hide length and zero 2419 wrong frames; "
        "c) most correct 2419-window frames, then highest re-acquisition at 80 frames; d) largest margin, then fewest components. "
        "Tie-break between K = 1 and K = 10 (mine, confirmed by the author as K = 10): higher total re-acquisition.",
        "6. make_lists_and_sheets.py, reference_track_sheet.py: review lists and contact sheets (sheets gitignored).",
        "7. bytetrack_reference.py: one untuned ByteTrack run (reference only).",
        "", "== Grids (sweep_lock_rule.GRID)",
        *(f"{k}: {['off' if v is None else v for v in vals]}" for k, vals in sw.GRID.items()),
        f"margin constants: failure centre distance {sw.FAIL_DIST_OVER_H} h, failure height ratio {sw.FAIL_HRATIO}, "
        f"failure gap {sw.FAIL_GAP} frames; legitimate bounds in legit_ranges.txt",
        "", "== Selection result", (OUT / "selection.txt").read_text().rstrip("\n").split("\n\n")[0] if (OUT / "selection.txt").exists() else "not run",
        "", "== Frame exclusion file (unchanged; the IMG_2419 entry stays until the author adopts the rule)",
        Path("validation/frame_exclusions.csv").read_text().rstrip("\n"),
        "", "== Code (SHA-256 of each file) and git state", f"git HEAD: {_git('rev-parse', 'HEAD')}",
        "uncommitted changes to these files when written: "
        + (_git("status", "--porcelain", "--", *scripts).replace("\n", "; ") or "none"),
        *(f"{s}: {sha256_file(Path(s))}" for s in scripts),
        "", "== Inputs (SHA-256)", f"box cache manifest: validation/box_cache/manifest.txt (clips, detector weights, versions)",
        *(f"{p.as_posix()}: {sha256_file(p)}" for p in inputs),
        "", "== Outputs (SHA-256)", *(f"{p.as_posix()}: {sha256_file(p)}" for p in outputs),
        "", "== Environment", *(f"{lib}: {_library_version(*(('opencv-python', 'opencv-python-headless') if lib == 'opencv-python' else (lib,)))}"
                               for lib in MANIFEST_LIBRARIES), ""]
    (OUT / "manifest.txt").write_text("\n".join(L))
    print(f"written {OUT / 'manifest.txt'}")


if __name__ == "__main__":
    main()
