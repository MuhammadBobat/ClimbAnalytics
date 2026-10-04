"""
Gym-specific hold colour -> grade-band mapping. See DECISIONS.md D-017, D-017a
and ARCHITECTURE.md §8.3.

GYM_COLOUR_GRADE_BANDS is real data, supplied directly by the author for their
own gym (2026-10-04) -- do not edit without updating DECISIONS.md D-017a.
Several entries are grade *bands*, not single values: colour alone doesn't
pin an exact grade at this gym, so downstream code must treat these as bands,
not fabricate a single number from them.

GYM_COLOUR_HSV_REFERENCE is a placeholder. Colour names are not HSV values --
the actual numeric ranges must be measured from real photos of this gym's
holds (lighting, exact shade, etc. all matter) via
training/calibrate_hold_colours.py. Per CLAUDE.md rule 9, do not invent these
numbers. Each entry stays None until calibrated, logged as part of D-017a.
"""

GYM_COLOUR_GRADE_BANDS: dict[str, str] = {
    "green": "V0",
    "white": "V0-V1",
    "blue": "V1-V3",
    "black": "V2-V4",
    "pink": "V2-V5",
    "red": "V3-V5",
    "purple": "V5-V7",
    "yellow": "V7-V8",
    "orange": "V8+",
}

# TBD, see DECISIONS.md D-017a -- calibrate via training/calibrate_hold_colours.py.
# Each value, once measured, should be ((h_min, h_max), (s_min, s_max), (v_min, v_max))
# in OpenCV HSV ranges (H: 0-179, S/V: 0-255).
GYM_COLOUR_HSV_REFERENCE: dict[str, tuple | None] = {
    colour: None for colour in GYM_COLOUR_GRADE_BANDS
}
