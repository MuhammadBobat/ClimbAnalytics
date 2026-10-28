# BlazePose z-value feasibility spike

**Exploratory spike. Not part of the pipeline. Not a logged decision.**

## Purpose

Check whether MediaPipe Pose (BlazePose GHUM)'s z-values give a usable
distance-to-wall-style signal on our existing bouldering footage. Nothing
here is adopted by the pipeline unless/until a real `DECISIONS.md` entry says
so. The outcome is logged as `DECISIONS.md` D-021: inconclusive (no ground
truth was available), not adopted.

Prior, already-logged context (read-only reference, not re-derived here):
`DECISIONS.md` D-016 already found MediaPipe Pose workable in this
environment once pinned to `mediapipe==0.10.35` (1.0.x hard-crashes on macOS
Apple Silicon). This spike reuses that pin, not a fresh discovery.

## Isolation

- Self-contained: own `requirements.txt`, own `.venv` (gitignored), own
  folder. No project runtime module is imported. No shared config, no
  `requirements.txt`/`pyproject` edits, nothing outside this folder touched
  except the project `.gitignore` (one line, for this folder's own large
  outputs).
- Branch: `spike/blazepose-z`, kept separate from `main` on purpose (pushed to
  the remote as a branch; not merged into `main`).
- 3D lifting (MotionBERT/MeTRAbs) is out of scope and not used here —
  BlazePose's z is a direct per-frame model output, not a lifting stage.

## How to run

```bash
cd experiments/blazepose_z_spike
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python run_spike.py       # runs BlazePose on the clips, writes output/
./.venv/bin/python analyze.py         # compares against the saved ViTPose-L CSVs
./.venv/bin/python make_overlays.py   # writes overlay videos to output/
```

The scripts take no arguments. `run_spike.py` downloads the BlazePose model into
`models/` on first use. Numbers and findings are in `REPORT.md`.

## Reproducibility

**This spike cannot be reproduced from a clean clone.** It reads inputs that are
gitignored and so are not in the repository:

- the source videos in `footage/test/` (`IMG_2412`, `IMG_2416`, `IMG_2417`,
  `IMG_2419`, `IMG_2420`), and
- the saved ViTPose-L keypoint CSVs in `validation/output/` (the 100-frame runs,
  written by `validation/validate_vitpose.py`).

The generated outputs (`output/`), the model file (`models/`) and the virtual
environment (`.venv/`) are also gitignored. To re-run it you need your own copies
of those inputs in place.

## Status

Finished. The spike was run on the five own-footage clips (first 100 frames each);
see `REPORT.md` for what was measured and `DECISIONS.md` D-021 for the logged
outcome (inconclusive, not adopted). Not developed further.
