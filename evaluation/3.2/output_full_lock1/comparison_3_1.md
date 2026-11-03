# Gap events of the 12 body joints: this run vs the 3.1 events on the same input

This run (a gap = consecutive frames with no usable position, any cause, detector-drop frames included): tier 1 142, tier 2 32, tier 3 4, total 178.
3.1 events on the same input (evaluation/3.1/output_full_lock1/events.csv; events end at detector-drop frames, which are counted separately): tier 1 134, tier 2 30, tier 3 4, total 168.

Differences are expected where a detector-drop frame touches a low-confidence run (merged here, split in 3.1) and from the whole-frame drops themselves, which are a gap for every body joint here and are not joint-level events in 3.1.
