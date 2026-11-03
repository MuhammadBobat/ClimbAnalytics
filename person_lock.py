"""
Single-climber lock rule (ARCHITECTURE.md §2; DECISIONS.md D-024, D-027 proposed).

Pure Python, no torch or cv2 imports, so pose_extraction.py can import it later. The legacy rule
(LockParams.legacy()) reproduces validation/validate_vitpose.py's select_person_box exactly: the box
nearest the previous centre, or the highest-confidence box when there is no lock yet. The gated
rule is OFF unless parameters are passed explicitly; nothing here sets project thresholds (D-027
sweeps them, CLAUDE.md rule 1).

Rule family (every part optional; None / off = not applied):
  G  distance gate: a box is a candidate only if its centre is within (gate_g0 + gate_v * gap) * h_ref
     of the last accepted centre, gap = consecutive frames without an accepted box before this frame.
  S  size consistency: height ratio to h_ref within [1/size_s, size_s].
  h_ref = median height of the last size_k accepted boxes (size_k = 1: the last accepted box).
  L  rescue tier: boxes with conf in [conf_rescue, conf_main) are considered only when no box with
     conf >= conf_main passes G and S.
  R  release: after release_n consecutive frames without an accepted box the gates are dropped and the
     highest-confidence box with conf >= conf_main is taken, as when there is no lock yet.
Selection among candidates that pass stays "nearest to the last accepted centre" (unchanged).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)


@dataclass(frozen=True)
class LockParams:
    conf_main: float = 0.5
    conf_rescue: float | None = None
    gate_g0: float | None = None
    gate_v: float = 0.0
    size_s: float | None = None
    size_k: int = 10
    release_n: int | None = None

    @classmethod
    def legacy(cls, conf_main: float = 0.5) -> "LockParams":
        return cls(conf_main=conf_main)

    def __post_init__(self) -> None:
        if self.conf_rescue is not None and not 0 < self.conf_rescue < self.conf_main:
            raise ValueError("conf_rescue must be in (0, conf_main)")
        if self.gate_g0 is not None and self.gate_g0 <= 0:
            raise ValueError("gate_g0 must be positive")
        if self.gate_v < 0 or (self.gate_g0 is None and self.gate_v != 0):
            raise ValueError("gate_v must be >= 0 and needs a gate")
        if self.size_s is not None and self.size_s <= 1:
            raise ValueError("size_s must be > 1")
        if self.size_k < 1:
            raise ValueError("size_k must be >= 1")
        if self.release_n is not None and self.release_n < 1:
            raise ValueError("release_n must be >= 1")

    @property
    def is_legacy(self) -> bool:
        return (self.conf_rescue is None and self.gate_g0 is None and self.size_s is None
                and self.release_n is None)

    def label(self) -> str:
        g = "off" if self.gate_g0 is None else f"{self.gate_g0:g}"
        s = "off" if self.size_s is None else f"{self.size_s:g}"
        c = "off" if self.conf_rescue is None else f"{self.conf_rescue:g}"
        n = "off" if self.release_n is None else f"{self.release_n}"
        return f"g0={g} v={self.gate_v:g} s={s} K={self.size_k} c={c} N={n}"


@dataclass(frozen=True)
class LockState:
    center: tuple[float, float] | None = None   # last accepted centre
    heights: tuple[float, ...] = ()              # heights of the last accepted boxes (newest last)
    gap: int = 0                                 # consecutive frames without an accepted box


@dataclass(frozen=True)
class LockInfo:
    tier: str            # "first" | "main" | "rescue" | "release" | "none"
    n_main: int          # boxes with conf >= conf_main this frame
    n_rescue: int        # boxes in [conf_rescue, conf_main)
    gap: int             # gap before this frame
    reason: str          # "accepted" | "no_box" | "gated"


def _reference_height(state: LockState, k: int) -> float:
    return statistics.median(state.heights[-k:])


def _passes(box: Box, state: LockState, params: LockParams) -> bool:
    if params.gate_g0 is None and params.size_s is None:
        return True
    h_ref = _reference_height(state, params.size_k)
    if params.gate_g0 is not None:
        cx, cy = box.center
        d = math.hypot(cx - state.center[0], cy - state.center[1])
        if d > (params.gate_g0 + params.gate_v * state.gap) * h_ref:
            return False
    if params.size_s is not None:
        r = box.h / h_ref
        if r > params.size_s or r < 1.0 / params.size_s:
            return False
    return True


def _nearest(boxes: list[Box], center: tuple[float, float]) -> Box:
    px, py = center
    return min(boxes, key=lambda b: (b.center[0] - px) ** 2 + (b.center[1] - py) ** 2)


def _accept(box: Box, state: LockState, params: LockParams, reset: bool = False) -> LockState:
    heights = (box.h,) if reset else (state.heights + (box.h,))[-max(params.size_k, 1):]
    return LockState(center=box.center, heights=heights, gap=0)


def select_climber_box(
    boxes: list[Box], state: LockState, params: LockParams
) -> tuple[Box | None, LockState, LockInfo]:
    """One frame: (chosen box or None, new state, info). `boxes` may contain any confidences; the
    tiers are cut by params.conf_main / conf_rescue."""
    main = [b for b in boxes if b.conf >= params.conf_main]
    rescue = ([b for b in boxes if params.conf_rescue <= b.conf < params.conf_main]
              if params.conf_rescue is not None else [])
    gap = state.gap
    info = lambda tier, reason: LockInfo(tier, len(main), len(rescue), gap, reason)  # noqa: E731

    if state.center is None:                       # no lock yet: highest-confidence box (legacy)
        if not main:
            return None, state, info("none", "no_box")
        box = max(main, key=lambda b: b.conf)
        return box, _accept(box, state, params, reset=True), info("first", "accepted")

    if params.release_n is not None and gap >= params.release_n and main:
        box = max(main, key=lambda b: b.conf)      # gates released after a long gap
        return box, _accept(box, state, params, reset=True), info("release", "accepted")

    ok = [b for b in main if _passes(b, state, params)]
    if ok:
        box = _nearest(ok, state.center)
        return box, _accept(box, state, params), info("main", "accepted")
    ok = [b for b in rescue if _passes(b, state, params)]
    if ok:
        box = _nearest(ok, state.center)
        return box, _accept(box, state, params), info("rescue", "accepted")
    reason = "gated" if (main or rescue) else "no_box"
    return None, LockState(state.center, state.heights, gap + 1), info("none", reason)


@dataclass(frozen=True)
class Rejection:
    """Why a gate or size check rejected the box the legacy rule would have chosen from the same state."""
    box: Box
    dist_over_h_ref: float      # centre distance to the last accepted centre, in units of h_ref
    allowance_over_h_ref: float | None   # gate_g0 + gate_v * gap, None if no gate
    h_ratio: float              # box height / h_ref
    gate_failed: bool
    size_failed: bool

    @property
    def check_failed(self) -> str:
        return "gate+size" if self.gate_failed and self.size_failed else ("gate" if self.gate_failed else "size")


def legacy_candidate(boxes: list[Box], state: LockState, params: LockParams) -> Box | None:
    """The box the LEGACY rule would choose from this state: the highest-confidence box with conf >= conf_main
    when there is no lock yet, else the one nearest the last accepted centre."""
    main = [b for b in boxes if b.conf >= params.conf_main]
    if not main:
        return None
    return max(main, key=lambda b: b.conf) if state.center is None else _nearest(main, state.center)


def explain_rejection(boxes: list[Box], state: LockState, params: LockParams) -> Rejection | None:
    """If the legacy rule's choice from this state fails the distance gate and/or the size check, say why
    (for the per-clip lock log); None if it passes, there is no lock, or no check is configured."""
    if state.center is None or (params.gate_g0 is None and params.size_s is None):
        return None
    cand = legacy_candidate(boxes, state, params)
    if cand is None:
        return None
    h_ref = _reference_height(state, params.size_k)
    d = math.hypot(cand.center[0] - state.center[0], cand.center[1] - state.center[1]) / h_ref
    allow = None if params.gate_g0 is None else params.gate_g0 + params.gate_v * state.gap
    ratio = cand.h / h_ref
    gate_failed = allow is not None and d > allow
    size_failed = params.size_s is not None and (ratio > params.size_s or ratio < 1.0 / params.size_s)
    if not (gate_failed or size_failed):
        return None
    return Rejection(cand, d, allow, ratio, gate_failed, size_failed)


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0
