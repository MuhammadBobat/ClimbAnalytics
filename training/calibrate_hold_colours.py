"""
Measure real HSV ranges for this gym's hold colours from sample photos, to
fill in hold_colours_config.py's GYM_COLOUR_HSV_REFERENCE placeholder. See
DECISIONS.md D-017, D-017a and ARCHITECTURE.md §8.3.

Per CLAUDE.md rule 9, these numbers must come from actually measuring real
photos, not be guessed -- this script exists specifically so nobody has to
guess them.

Directory convention: put a handful of cropped photos of known-colour holds
under training/colour_samples/<colour_name>/*.jpg, one subfolder per colour
name exactly matching hold_colours_config.py's GYM_COLOUR_GRADE_BANDS keys
(green, white, blue, black, pink, red, purple, yellow, orange). Crop tightly
to the hold itself -- background/wall pixels will skew the measurement.

Usage:
    python training/calibrate_hold_colours.py
    python training/calibrate_hold_colours.py --samples-dir training/colour_samples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from hold_colours_config import GYM_COLOUR_GRADE_BANDS  # noqa: E402


def measure_colour(image_paths: list[Path]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    all_h, all_s, all_v = [], [], []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  WARNING: could not read {path}, skipping")
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        all_h.extend(hsv[:, 0])
        all_s.extend(hsv[:, 1])
        all_v.extend(hsv[:, 2])
    if not all_h:
        return None
    h_arr, s_arr, v_arr = np.array(all_h), np.array(all_s), np.array(all_v)
    # 10th-90th percentile range rather than min/max, to avoid single noisy pixels
    return (
        (float(np.percentile(h_arr, 10)), float(np.percentile(h_arr, 90))),
        (float(np.percentile(s_arr, 10)), float(np.percentile(s_arr, 90))),
        (float(np.percentile(v_arr, 10)), float(np.percentile(v_arr, 90))),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples-dir", type=Path, default=Path("training/colour_samples"))
    args = parser.parse_args()

    if not args.samples_dir.exists():
        print(f"{args.samples_dir} does not exist yet.")
        print("Create one subfolder per colour (matching hold_colours_config.py's "
              f"keys: {', '.join(GYM_COLOUR_GRADE_BANDS)}), each with a few cropped "
              "sample photos of that colour's holds, then re-run this script.")
        sys.exit(0)

    results = {}
    for colour in GYM_COLOUR_GRADE_BANDS:
        colour_dir = args.samples_dir / colour
        if not colour_dir.exists():
            print(f"{colour}: no sample folder at {colour_dir}, skipping")
            continue
        image_paths = sorted(p for p in colour_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not image_paths:
            print(f"{colour}: no images in {colour_dir}, skipping")
            continue
        measured = measure_colour(image_paths)
        if measured:
            results[colour] = measured
            print(f"{colour}: measured from {len(image_paths)} image(s)")
        else:
            print(f"{colour}: could not read any images in {colour_dir}")

    if not results:
        print("\nNo colours measured. Add sample photos first.")
        return

    print("\nPaste this into hold_colours_config.py's GYM_COLOUR_HSV_REFERENCE "
          "(for the colours not shown, keep them None until you add samples):\n")
    for colour, (h_range, s_range, v_range) in results.items():
        print(f'    "{colour}": (({h_range[0]:.0f}, {h_range[1]:.0f}), '
              f'({s_range[0]:.0f}, {s_range[1]:.0f}), ({v_range[0]:.0f}, {v_range[1]:.0f})),')
    print(f"\nLog this measurement as DECISIONS.md D-017a once applied "
          f"(number of sample photos per colour, date measured).")


if __name__ == "__main__":
    main()
