"""
Tests for the machine-readable outputs of validation/analyze_occlusion_gaps.py
(events.csv, summary.csv, frame_losses.csv, manifest.txt). Synthetic, hand-computed.

Scenario (clip s.MOV, 12 frames): frame 5 dropped by the detector; frames 8-9 excluded
by list; left_wrist low-confidence on 1-2 and 6-7; nose low-confidence on 0.
  used frames = 12 - 1 - 2 = 9.

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from validation import analyze_occlusion_gaps as a

REPO_ROOT = Path(__file__).resolve().parents[1]
SHA256_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def write_clip(path, detected, low):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame_idx", "timestamp_s", "keypoint_id", "keypoint_name",
                    "x", "y", "confidence"])
        for fr in detected:
            for i, n in enumerate(a.COCO17_NAMES):
                w.writerow(["s.MOV", fr, f"{fr / 30:.6f}", i, n, 0, 0,
                            0.2 if (fr, n) in low else 0.9])


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        low = {(f, "left_wrist") for f in (1, 2, 6, 7)} | {(0, "nose")}
        p = self.d / "s_keypoints.csv"
        write_clip(p, [f for f in range(12) if f != 5], low)
        self.stats = [a.analyze_clip(p, 12, excluded={8, 9})]

    def test_events_csv(self):
        out = self.d / "events.csv"
        a.write_events_csv(out, self.stats)
        rows = read_csv(out)
        self.assertEqual(list(rows[0].keys()), a.EVENTS_COLUMNS)
        got = [(r["joint"], r["first_frame"], r["last_frame"], r["length_frames"],
                r["touches_clip_start_or_end"], r["adjacent_to_dropped_or_excluded"],
                r["pipeline_joint"]) for r in rows]
        self.assertEqual(got, [
            ("nose", "0", "0", "1", "True", "False", "False"),
            ("left_wrist", "1", "2", "2", "False", "False", "True"),
            ("left_wrist", "6", "7", "2", "False", "True", "True"),
        ])

    def test_frame_losses_csv(self):
        out = self.d / "fl.csv"
        a.write_frame_losses_csv(out, self.stats)
        got = [(r["kind"], r["first_frame"], r["last_frame"], r["length_frames"])
               for r in read_csv(out)]
        self.assertEqual(got, [("detector_drop", "5", "5", "1"), ("excluded_by_list", "8", "9", "2")])

    def test_summary_csv_values_and_denominators(self):
        out = self.d / "summary.csv"
        a.write_summary_csv(out, self.stats)
        rows = {(r["clip"], r["joint"]): r for r in read_csv(out)}
        self.assertEqual(list(next(iter(rows.values())).keys()), a.SUMMARY_COLUMNS)

        w = rows[("s.MOV", "left_wrist")]
        self.assertEqual((w["unit"], w["n_events"], w["n_lt5"], w["frames_total"],
                          w["frames_detector_dropped"], w["frames_excluded_by_list"],
                          w["frames_used"], w["frames_low_conf"]),
                         ("frames", "2", "2", "12", "1", "2", "9", "4"))
        self.assertEqual(w["pct_low_conf_of_used"], "44.444")        # 4/9
        self.assertEqual(w["pct_unavailable_of_total"], "58.333")     # (4+1+2)/12

        g = rows[("s.MOV", "PIPELINE_JOINTS")]
        self.assertEqual((g["unit"], g["pipeline_joint"], g["n_events"], g["frames_total"],
                          g["frames_detector_dropped"], g["frames_excluded_by_list"],
                          g["frames_used"], g["frames_low_conf"]),
                         ("joint-frames", "True", "2", "96", "8", "16", "72", "4"))
        self.assertEqual(g["pct_low_conf_of_used"], "5.556")          # 4/72
        self.assertEqual(g["pct_unavailable_of_total"], "29.167")     # (4+8+16)/96

        al = rows[("s.MOV", "ALL_JOINTS")]
        self.assertEqual((al["n_events"], al["frames_total"], al["frames_used"], al["frames_low_conf"]),
                         ("3", "204", "153", "5"))
        # ALL clip rows equal the single clip's
        self.assertEqual(rows[("ALL", "left_wrist")]["frames_low_conf"], "4")

    def test_total_equals_dropped_plus_excluded_plus_used(self):
        out = self.d / "summary.csv"
        a.write_summary_csv(out, self.stats)
        for r in read_csv(out):
            self.assertEqual(int(r["frames_total"]),
                             int(r["frames_detector_dropped"]) + int(r["frames_excluded_by_list"])
                             + int(r["frames_used"]))

    def test_outputs_are_deterministic(self):
        for name, fn in [("events.csv", a.write_events_csv), ("summary.csv", a.write_summary_csv),
                         ("frame_losses.csv", a.write_frame_losses_csv)]:
            fn(self.d / ("1_" + name), self.stats)
            fn(self.d / ("2_" + name), self.stats)
            self.assertEqual((self.d / ("1_" + name)).read_bytes(), (self.d / ("2_" + name)).read_bytes())

    def test_pipeline_section_in_markdown(self):
        text = a.render_report(self.stats)
        self.assertIn("## 4. Pipeline joints (8)", text)
        self.assertIn("left_wrist: 2 event(s)", text)


class ManifestTests(unittest.TestCase):
    def test_manifest_contents(self):
        d = Path(tempfile.mkdtemp())
        footage = d / "footage"
        footage.mkdir()
        (footage / "s.MOV").write_bytes(b"abc")
        excl = d / "ex.csv"
        excl_text = "# note\nvideo,start_frame,end_frame,reason\ns.MOV,8,9,test reason\n"
        excl.write_text(excl_text)
        p = d / "s_keypoints.csv"
        write_clip(p, range(12), set())
        stats = [a.analyze_clip(p, 12, excluded={8, 9})]
        text = a.build_manifest(stats, d, footage, excl, today="2026-10-28")
        self.assertIn("date: 2026-10-28", text)
        self.assertIn(f"occlusion confidence threshold: {a.OCCLUSION_CONFIDENCE_THRESHOLD}", text)
        self.assertIn(a.VITPOSE_MODEL_NAME, text)
        self.assertIn(f"dataset_index: {a.VITPOSE_DATASET_INDEX}", text)
        self.assertIn(excl_text.rstrip("\n"), text)  # exclusion contents verbatim
        self.assertIn(f"s.MOV: frames 12", text)
        self.assertIn(f"sha256 {SHA256_ABC}", text)
        self.assertIn("git HEAD:", text)
        for lib in a.MANIFEST_LIBRARIES:
            self.assertIn(f"{lib}:", text)

    def test_missing_video_and_exclusions_are_stated(self):
        d = Path(tempfile.mkdtemp())
        p = d / "s_keypoints.csv"
        write_clip(p, range(4), set())
        stats = [a.analyze_clip(p, 4)]
        text = a.build_manifest(stats, d, d / "nofootage", d / "none.csv", today="x")
        self.assertIn("video not found", text)
        self.assertIn("== Frame exclusion file\nnone", text)

    def test_sha256_known_value(self):
        f = Path(tempfile.mkdtemp()) / "x"
        f.write_bytes(b"abc")
        self.assertEqual(a.sha256_file(f), SHA256_ABC)


class ConstantsTests(unittest.TestCase):
    def test_pipeline_joints(self):
        self.assertEqual(
            sorted(a.PIPELINE_JOINTS),
            sorted(["left_shoulder", "right_shoulder", "left_hip", "right_hip",
                    "left_wrist", "right_wrist", "left_ankle", "right_ankle"]),
        )

    def test_model_constants_match_validate_vitpose(self):
        text = (REPO_ROOT / "validation" / "validate_vitpose.py").read_text()
        self.assertIn(f'"{a.VITPOSE_MODEL_NAME}"', text)
        self.assertIn(f"VITPOSE_COCO_DATASET_INDEX = {a.VITPOSE_DATASET_INDEX}", text)
        self.assertIn(Path(a.DETECTOR_WEIGHTS).name, text)
        self.assertIn('weights_dir = Path(__file__).parent / "weights"', text)


class RealDataRecountTest(unittest.TestCase):
    """Independent recount of the number of gap events from the raw CSVs."""

    FULL = REPO_ROOT / "validation" / "output_full"

    def test_event_count_matches_independent_recount(self):
        if not (self.FULL / "summary.csv").exists():
            msg = ("SKIPPED real-data recount: validation/output_full/ not found "
                   "(gitignored; generate it with validate_vitpose.py --output-dir validation/output_full)")
            print(msg, file=sys.stderr)
            self.skipTest(msg)
        excl = a.load_exclusions(REPO_ROOT / "validation" / "frame_exclusions.csv")
        totals = {r["video"]: int(r["total_frames"]) for r in read_csv(self.FULL / "summary.csv")}
        stats, expected = [], 0
        for p in sorted(self.FULL.glob("*_keypoints.csv")):
            rows = read_csv(p)
            clip = rows[0]["video"]
            stats.append(a.analyze_clip(p, totals[clip], excl.get(clip)))
            used = {int(r["frame_idx"]) for r in rows} - excl.get(clip, set())
            conf = {(r["keypoint_name"], int(r["frame_idx"])): float(r["confidence"]) for r in rows}
            for j in a.COCO17_NAMES:
                prev = False
                for f in range(totals[clip]):
                    occ = f in used and conf[(j, f)] < 0.5
                    if occ and not prev:
                        expected += 1
                    prev = occ
        out = Path(tempfile.mkdtemp()) / "events.csv"
        a.write_events_csv(out, stats)
        self.assertEqual(len(read_csv(out)), expected)


if __name__ == "__main__":
    unittest.main()
