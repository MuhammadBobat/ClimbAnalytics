"""
Tests for the --lock-rule option of validation/validate_vitpose.py (DECISIONS.md D-027): the gated rule is the default
(adopted 2026-11-03), the legacy rule stays available and unchanged, GATED_LOCK_PARAMS are pinned, and process_video
writes the documented files in each mode. The detector and the pose model are replaced by scripted fakes; nothing here
is a measurement and no real model is loaded.

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import csv
import importlib.util
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

from person_lock import Box, LockParams, LockState, select_climber_box

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("validate_vitpose", REPO_ROOT / "validation" / "validate_vitpose.py")
vv = importlib.util.module_from_spec(_spec)
sys.modules["validate_vitpose"] = vv
_spec.loader.exec_module(vv)

CLIMBER = vv.PersonBox(60, 20, 30, 100, 0.9)       # centre (75, 70), h 100
BYSTANDER = vv.PersonBox(250, 150, 12, 25, 0.7)     # far and small
CLIMBER_WEAK = vv.PersonBox(62, 22, 30, 98, 0.3)    # a rescue-tier box on the climber

# per frame: the boxes the fake detector returns when asked for confidence >= 0.1
SCRIPT = [[CLIMBER], [CLIMBER], [], [BYSTANDER, CLIMBER_WEAK], [BYSTANDER, CLIMBER_WEAK], [BYSTANDER, CLIMBER]]


def make_video(path: Path, n: int) -> None:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    for _ in range(n):
        w.write(np.zeros((240, 320, 3), dtype=np.uint8))
    w.release()


class FakeDetector:
    """Replaces detect_person_boxes: returns the scripted boxes with conf >= the requested threshold."""
    def __init__(self):
        self.i = 0

    def __call__(self, detector, frame, conf_threshold, device):
        boxes = [b for b in SCRIPT[self.i] if b.conf >= conf_threshold]
        self.i += 1
        return boxes


def fake_run_pose(pose_processor, pose_model, image, box, device):
    return np.zeros((17, 2)) + [box.x, box.y], np.full(17, 0.9)


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class ParserTests(unittest.TestCase):
    def test_gated_is_the_default_and_legacy_is_accepted(self):
        p = vv.build_parser()
        self.assertEqual(p.parse_args([]).lock_rule, "gated")
        self.assertEqual(p.parse_args(["--lock-rule", "legacy"]).lock_rule, "legacy")
        self.assertEqual(p.parse_args(["--lock-rule", "gated"]).lock_rule, "gated")
        with self.assertRaises(SystemExit):
            p.parse_args(["--lock-rule", "other"])

    def test_gated_parameters_are_pinned(self):
        self.assertEqual(vv.GATED_LOCK_PARAMS, dict(gate_g0=0.25, gate_v=0.0, size_s=1.75, size_k=10,
                                                    conf_rescue=0.1, release_n=None))
        LockParams(conf_main=0.5, **vv.GATED_LOCK_PARAMS)      # valid with the default --bbox-conf

    def test_legacy_rule_is_unchanged(self):
        # select_person_box equals person_lock's legacy parameters on a scripted sequence
        prev, st = None, LockState()
        for boxes in SCRIPT:
            main = [b for b in boxes if b.conf >= 0.5]
            want = vv.select_person_box(main, prev)
            got, st, _ = select_climber_box([Box(b.x, b.y, b.w, b.h, b.conf) for b in main], st, LockParams.legacy())
            self.assertEqual(None if want is None else (want.x, want.y, want.w, want.h),
                             None if got is None else (got.x, got.y, got.w, got.h))
            if want is not None:
                prev = want.center


class ProcessVideoTests(unittest.TestCase):
    def run_mode(self, mode):
        d = Path(tempfile.mkdtemp())
        video = d / "clip.mp4"
        make_video(video, len(SCRIPT))
        out = d / "out"
        out.mkdir()
        saved = vv.detect_person_boxes, vv.run_pose
        vv.detect_person_boxes, vv.run_pose = FakeDetector(), fake_run_pose
        try:
            vv.process_video(video, out, None, None, None, "cpu",
                             Namespace(max_frames=None, bbox_conf=0.5, lock_rule=mode))
        finally:
            vv.detect_person_boxes, vv.run_pose = saved
        return out

    def boxes_by_frame(self, out):
        res = {}
        for r in read(out / "clip_keypoints.csv"):
            res.setdefault(int(r["frame_idx"]), (float(r["bbox_x"]), float(r["bbox_w"]), int(r["n_person_boxes"])))
        return res

    def test_legacy_writes_only_the_original_files_and_follows_the_bystander(self):
        out = self.run_mode("legacy")
        self.assertEqual(sorted(p.name for p in out.iterdir()),
                         ["clip_flags.csv", "clip_keypoints.csv", "clip_overlay.mp4"])
        b = self.boxes_by_frame(out)
        self.assertEqual(b[3][0], 250)           # the bystander is the only box >= 0.5: legacy accepts it
        self.assertEqual(b[4][0], 250)
        self.assertEqual(b[5][0], 250)           # and keeps it even when the climber is back (the D-024 failure)
        self.assertEqual(set(b) , {0, 1, 3, 4, 5})   # frame 2 has no box
        header = open(out / "clip_keypoints.csv").readline().strip().split(",")
        self.assertEqual(header[-1], "n_person_boxes")

    def test_gated_keeps_the_climber_and_logs_the_rejection(self):
        out = self.run_mode("gated")
        self.assertEqual(sorted(p.name for p in out.iterdir()),
                         ["clip_flags.csv", "clip_keypoints.csv", "clip_lock.csv", "clip_lock_frames.csv",
                          "clip_overlay.mp4"])
        b = self.boxes_by_frame(out)
        self.assertEqual(b[3][0], 62)            # the weak box on the climber, accepted through the rescue tier
        self.assertEqual(b[4][0], 62)
        self.assertEqual(b[5][0], 60)
        self.assertNotIn(2, b)                   # a frame with no box at all stays dropped
        self.assertEqual(b[3][2], 1)             # n_person_boxes still counts boxes >= --bbox-conf (the bystander)
        header = open(out / "clip_keypoints.csv").readline().strip().split(",")
        self.assertEqual(header[-1], "n_person_boxes")
        tiers = {int(r["frame_idx"]): r["tier"] for r in read(out / "clip_lock_frames.csv")}
        self.assertEqual([tiers[i] for i in range(6)], ["first", "main", "none", "rescue", "rescue", "main"])
        rej = read(out / "clip_lock.csv")
        self.assertEqual([int(r["frame_idx"]) for r in rej], [3, 4])
        self.assertEqual(rej[0]["check_failed"], "gate+size")
        self.assertEqual(rej[0]["rule_outcome"], "rescue")
        self.assertEqual(float(rej[0]["rejected_x"]), 250.0)

    def test_both_modes_agree_where_nothing_is_rejected(self):
        leg, gat = self.boxes_by_frame(self.run_mode("legacy")), self.boxes_by_frame(self.run_mode("gated"))
        for t in (0, 1):
            self.assertEqual(leg[t], gat[t])


if __name__ == "__main__":
    unittest.main()
