"""
Contact sheet for confirming the PROVISIONAL IMG_2419 reference track (reference_track_2419.csv, frames 592-614;
DECISIONS.md D-024, D-027): the reference box in blue with its detector confidence, the legacy choice in red
(the bystander from 597), frames 591 and 615 for context. Output: evaluation/lock_rule/sheets/IMG_2419_reference_track.png
(gitignored).

Usage:
    python3 evaluation/lock_rule/reference_track_sheet.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lock_common import LockParams, load_cache, reference_track_2419, run_rule  # noqa: E402
from make_lists_and_sheets import read_frames  # noqa: E402

SCALE, COLS = 1 / 3, 6


def main() -> None:
    import cv2
    import numpy as np
    frames = load_cache("IMG_2419")
    legacy = run_rule(frames, LockParams.legacy())[0]
    ref = reference_track_2419(frames, legacy)
    idxs = list(range(591, 616))
    imgs = read_frames(Path("footage/test/IMG_2419.MOV"), idxs)
    tiles = []
    for t in idxs:
        im = cv2.resize(imgs[t], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        for box, col, name in ((legacy[t], (0, 0, 255), "legacy"), (ref.get(t, (None, 0))[0], (255, 120, 0), "ref")):
            if box is None:
                continue
            p1, p2 = (int(box.x * SCALE), int(box.y * SCALE)), (int((box.x + box.w) * SCALE), int((box.y + box.h) * SCALE))
            if name == "ref":
                p1, p2 = (p1[0] + 3, p1[1] + 3), (p2[0] - 3, p2[1] - 3)
            cv2.rectangle(im, p1, p2, col, 2)
            if name == "ref":
                cv2.putText(im, f"{box.conf:.2f}", (p1[0] + 4, p1[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        inref = t in ref
        cv2.rectangle(im, (0, 0), (im.shape[1], 20), (30, 30, 30), -1)
        cv2.putText(im, f"f{t}" + ("" if inref else " (context)"), (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(im)
    h, w = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / COLS)
    canvas = np.full((28 + rows * h, COLS * w, 3), 20, dtype=np.uint8)
    cv2.putText(canvas, "IMG_2419 reference track 592-614: BLUE = reference box (with detector confidence), RED = legacy choice",
                (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    for i, tile in enumerate(tiles):
        r, k = divmod(i, COLS)
        canvas[28 + r * h:28 + (r + 1) * h, k * w:(k + 1) * w] = tile
    out = Path("evaluation/lock_rule/sheets/IMG_2419_reference_track.png")
    cv2.imwrite(str(out), canvas)
    print(out, canvas.shape)


if __name__ == "__main__":
    main()
