"""
ppe_monitor — Personal Protective Equipment (helmet & vest) monitoring for construction sites.

Same logic as the Snehil_PPE_V4 notebook (= V2 + TensorRT + display-only skeleton), packaged
as a standalone module. Weights and TensorRT engines are built once and cached on Google
Drive; subsequent runs only load them.

Usage (Colab):
    import ppe_monitor as pm
    pm.init()                          # first run: download weights + build engines (~10 min). Then: seconds
    pm.analyze("/content/drive/MyDrive/ppe_monitor/inputs/3.mp4")
    pm.analyze("photo.jpg")

Architecture (details in docs/ARCHITECTURE.md):
    A) Person: YOLO11n (COCO) + ByteTrack
    B) PPE: Snehil YOLOv8n, only Hardhat / NO-Hardhat / Safety Vest / NO-Safety Vest
    C) Association: helmet only on the head region, vest only on the torso (box geometry)
    D) Per-ID temporal memory + hysteresis
    Display: HUD + skeleton (pose is drawing-only; it plays no part in any decision)
"""
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import os, time, math, logging, shutil, subprocess, urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import cv2
import torch
from PIL import Image
import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.utils import LOGGER

try:
    from IPython.display import display, Video
except ImportError:                                   # outside Jupyter
    display = print
    Video = None

try:
    import arabic_reshaper
    try:
        from bidi.algorithm import get_display
    except ImportError:
        from bidi import get_display

    def fa(text):
        """Prepare Persian/mixed text for matplotlib (join letters + right-to-left)."""
        return get_display(arabic_reshaper.reshape(str(text)))
except ImportError:
    def fa(text):
        return str(text)

__version__ = "4.3.0"

# ============================================================
# Paths & hardware
# ============================================================
DEFAULT_BASE_DIR = Path("/content/drive/MyDrive/ppe_monitor")
BASE_DIR = WORK_DIR = WEIGHTS_DIR = ENGINE_DIR = OUTPUT_DIR = EVENTS_DIR = INPUT_DIR = None

DEVICE = 0 if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    FP16_KW = {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}
else:
    FP16_KW = {}

SNEHIL_REPO = "https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection"
# Self-contained first: every weight is shipped in this repository's weights/ folder.
REPO_RAW = "https://github.com/ParsaVictor/PPE_detection-Sentinel/raw/main/weights"
SNEHIL_BEST_URLS = [f"{REPO_RAW}/snehil_best.pt",
                    f"{SNEHIL_REPO}/raw/main/models/best.pt",
                    f"{SNEHIL_REPO}/raw/master/models/best.pt"]

PPE_CLASS_NAMES = ["Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest"]
SLOT_OF = {
    "Hardhat":        ("helmet", True),
    "NO-Hardhat":     ("helmet", False),
    "Safety Vest":    ("vest",   True),
    "NO-Safety Vest": ("vest",   False),
}

# These are populated in init()
ppe_model = ppe_crop_model = person_model = pose_model = pipe = None
PPE_BACKEND = PERSON_BACKEND = POSE_BACKEND = "off"
PERSON_WEIGHTS = BEST_PT = None
PPE_NAME_TO_ID, PPE_CLASS_IDS = {}, []
_READY = False


# ============================================================
# Config
# ============================================================


# ============================================================
# Section 2) Config — the only place where the numbers are set
# ============================================================
SLOTS = ("helmet", "vest")


@dataclass
class Config:
    # ---------- Stage A — person (COCO) ----------
    person_model: str = "yolo11n.pt"   # for very distant people: "yolo11s.pt"
    person_imgsz: int = 640
    person_conf: float = 0.20
    person_iou: float = 0.50
    person_min_height_px: int = 24
    person_display_conf: float = 0.35  # image display only: a weak person with no PPE is not drawn
    edge_margin_px: int = 3
    tracker: str = "bytetrack.yaml"    # "botsort.yaml" = more accurate in crowds, heavier

    # ---------- Stage B — PPE (Snehil) ----------
    ppe_imgsz: int = 640
    ppe_raw_conf: float = 0.15         # low; the real threshold is min_conf in Stage C
    ppe_iou: float = 0.70             # same as the Ultralytics default (identical to the base model)
    ppe_every_n_frames: int = 1        # 2 = run the PPE model every other frame
    skip_ppe_without_person: bool = True

    crop_refine: bool = True
    crop_imgsz: int = 320
    crop_min_person_h: int = 40
    crop_max_person_h: int = 360
    crop_max_per_frame: int = 4
    crop_cooldown_frames: int = 5      # new: at most one crop per person every 5 frames
    crop_pad_x: float = 0.15
    crop_pad_top: float = 0.20
    crop_verify_violations: bool = True
    crop_skip_confident: float = 2.0

    # ---------- Stage C — Association ----------
    min_conf: dict = field(default_factory=lambda: {
        "Hardhat": 0.30, "NO-Hardhat": 0.35,
        "Safety Vest": 0.30, "NO-Safety Vest": 0.35,
    })
    min_ioa: float = 0.40
    x_margin: float = 0.15
    top_margin: float = 0.20
    helmet_band: tuple = (-0.20, 0.40)
    helmet_ideal: float = 0.08
    helmet_max_w_frac: float = 1.30
    helmet_max_h_frac: float = 0.45   # relative to the "reference height" (below) — not the box height
    full_body_aspect: float = 2.0      # reference height = max(box height, 2.0 × box width) — for bent-over / half-body people
    vest_band: tuple = (0.10, 0.85)
    vest_ideal: float = 0.42
    vest_min_w_frac: float = 0.20
    vest_min_h_frac: float = 0.10
    vest_max_h_frac: float = 0.85
    vest_min_visible_aspect: float = 1.3   # new: close-up view (head & shoulders only) → the vest is not judged
    conflict_margin: float = 0.08
    required_ppe: tuple = ("helmet", "vest")

    # ---------- Stage D — temporal memory ----------
    decay: float = 0.85
    s_max: float = 3.0
    on_ok: float = 1.0
    on_violation: float = 1.5
    min_track_hits: int = 5
    stale_frames: int = 45
    forget_frames: int = 150
    absence_evidence: float = 0.0
    absence_min_person_h: int = 140
    event_min_frames: int = 8

    # ---------- Backend (new in v4) ----------
    backend: str = "tensorrt"          # "tensorrt" | "pytorch" — automatic PyTorch fallback if TensorRT fails

    # ---------- Skeleton — display only (new in v4) ----------
    show_skeleton: bool = True         # False = the pose model never runs at all
    pose_model: str = "yolo11n-pose.pt"
    kpt_conf: float = 0.50             # only keypoints the model is confident about are drawn

    # ---------- Display ----------
    hold_frames: int = 10              # if the model misses a helmet/vest in one frame, the last box stays on the person for up to 10 frames
    show_fps: bool = True
    display_full_box: bool = True      # (a) if the winner came from a crop, draw the (larger) overlapping full-frame box
    display_match_iou: float = 0.30
    show_not_judgeable: bool = True    # (d) "not judgeable" (e.g. vest in a close-up) is drawn in gray
    debug_show_rejected: bool = False
    debug_show_regions: bool = False
    fp16: bool = torch.cuda.is_available()


cfg = Config()


# ============================================================
# Section 4) Data structures + Stage A (person) + Stage B (PPE)
# ============================================================
@dataclass
class Det:
    xyxy: np.ndarray
    conf: float
    cls_name: str
    slot: str
    positive: bool
    source: str = "full"        # "full" | "crop"
    owner: int = -1
    assoc: float = 0.0
    status: str = "pending"     # accepted | rejected | superseded
    reason: str = ""

    @property
    def evidence(self):
        return self.conf * self.assoc


@dataclass
class Person:
    xyxy: np.ndarray
    conf: float
    track_id: object = None
    top_cut: bool = False
    bottom_cut: bool = False
    kpts: object = None           # body keypoints — skeleton drawing only
    frame: dict = field(default_factory=dict)    # slot -> (decision, evidence, winning Det) — this frame
    status: dict = field(default_factory=dict)   # slot -> yes / no / unknown — final
    score: dict = field(default_factory=dict)

    @property
    def w(self):
        return float(self.xyxy[2] - self.xyxy[0])

    @property
    def h(self):
        return float(self.xyxy[3] - self.xyxy[1])

    def overall(self, required):
        states = [self.status.get(s, "unknown") for s in required]
        if "no" in states:
            return "violation"
        if all(s == "yes" for s in states):
            return "ok"
        return "unknown"


def _model_ms(result):
    return sum(v for v in (result.speed or {}).values() if v is not None)


def detect_persons(frame, tracker_model=None):
    """Stage A → (persons, the model's own time in ms)."""
    kw = dict(imgsz=cfg.person_imgsz, conf=cfg.person_conf, iou=cfg.person_iou,
              classes=[0], device=DEVICE, verbose=False, **run_kw(PERSON_BACKEND))
    if tracker_model is not None:
        r = tracker_model.track(frame, persist=True, tracker=cfg.tracker, **kw)[0]
    else:
        r = person_model.predict(frame, **kw)[0]

    b = r.boxes
    if b is None or len(b) == 0:
        return [], _model_ms(r)
    H = frame.shape[0]
    xyxy = b.xyxy.cpu().numpy().astype(np.float32)
    conf = b.conf.cpu().numpy()
    ids = b.id.cpu().numpy().astype(int).tolist() if b.id is not None else [None] * len(conf)

    persons = []
    for box, c, tid in zip(xyxy, conf, ids):
        if box[3] - box[1] < cfg.person_min_height_px:
            continue
        persons.append(Person(xyxy=box, conf=float(c), track_id=tid,
                              top_cut=bool(box[1] <= cfg.edge_margin_px),
                              bottom_cut=bool(box[3] >= H - cfg.edge_margin_px)))
    return persons, _model_ms(r)


def _to_dets(result, ox=0, oy=0, source="full"):
    b = result.boxes
    if b is None or len(b) == 0:
        return []
    xyxy = b.xyxy.cpu().numpy().astype(np.float32) + np.array([ox, oy, ox, oy], dtype=np.float32)
    conf = b.conf.cpu().numpy()
    cls = b.cls.cpu().numpy().astype(int)
    dets = []
    for box, c, k in zip(xyxy, conf, cls):
        name = result.names[k]
        if name in SLOT_OF:
            slot, positive = SLOT_OF[name]
            dets.append(Det(xyxy=box, conf=float(c), cls_name=name, slot=slot, positive=positive, source=source))
    return dets


def _ppe_kw(imgsz, backend):
    return dict(imgsz=imgsz, conf=cfg.ppe_raw_conf, iou=cfg.ppe_iou, classes=PPE_CLASS_IDS,
                device=DEVICE, verbose=False, **run_kw(backend))


def detect_ppe_full(frame):
    """Stage B1 → (dets, the model's own time in ms)."""
    r = ppe_model.predict(frame, **_ppe_kw(cfg.ppe_imgsz, PPE_BACKEND))[0]
    return _to_dets(r), _model_ms(r)


def person_crop_window(p, H, W):
    x1, y1, x2, y2 = p.xyxy
    return (int(max(0, x1 - cfg.crop_pad_x * p.w)),
            int(max(0, y1 - cfg.crop_pad_top * p.h)),
            int(min(W, x2 + cfg.crop_pad_x * p.w)),
            int(min(H, y2 + 0.05 * p.h)))


def detect_ppe_crops(frame, persons, idxs):
    """Stage B2 — crops of the selected people, all in one batch."""
    H, W = frame.shape[:2]
    crops, offsets = [], []
    for i in idxs:
        cx1, cy1, cx2, cy2 = person_crop_window(persons[i], H, W)
        if cx2 - cx1 >= 8 and cy2 - cy1 >= 8:
            crops.append(np.ascontiguousarray(frame[cy1:cy2, cx1:cx2]))
            offsets.append((cx1, cy1))
    if not crops:
        return []
    dets = []
    for r, (ox, oy) in zip(ppe_crop_model.predict(crops, **_ppe_kw(cfg.crop_imgsz, "pytorch")), offsets):
        dets += _to_dets(r, ox, oy, source="crop")
    return dets


def attach_skeletons(frame, persons):
    """
    Display only: the pose model runs on the frame and each skeleton's keypoints are given to
    the person with the largest overlap (IoU >= 0.4). No helmet/vest decision depends on this.
    """
    if pose_model is None or not persons:
        return 0.0
    r = pose_model.predict(frame, imgsz=cfg.person_imgsz, conf=0.25, classes=[0], device=DEVICE,
                           verbose=False, **run_kw(POSE_BACKEND))[0]
    if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
        return _model_ms(r)
    boxes = r.boxes.xyxy.cpu().numpy()
    kps = r.keypoints.data.cpu().numpy()
    used = set()
    for p in persons:
        best_j, best = -1, 0.4
        for j, b in enumerate(boxes):
            if j in used:
                continue
            ix = max(0.0, min(p.xyxy[2], b[2]) - max(p.xyxy[0], b[0]))
            iy = max(0.0, min(p.xyxy[3], b[3]) - max(p.xyxy[1], b[1]))
            inter = ix * iy
            iou = inter / (p.w * p.h + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)
            if iou > best:
                best_j, best = j, iou
        if best_j >= 0:
            used.add(best_j)
            p.kpts = kps[best_j]
    return _model_ms(r)


# ============================================================
# Section 5) Stage C — geometric PPE ↔ Person association
# ============================================================
REASON_PRIORITY = ["bad_size", "not_on_head", "not_on_torso", "outside_person"]


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _ioa(inner, outer):
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1) / (_area(inner) + 1e-6)


def slot_band(p, slot):
    if slot == "helmet":
        lo, hi = cfg.helmet_band
        return lo, hi, cfg.helmet_ideal
    lo, hi = cfg.vest_band
    if p.bottom_cut:
        hi = max(hi, 0.95)
    return lo, hi, cfg.vest_ideal


def torso_visible(p):
    """close-up (only head & shoulders in frame) → the torso cannot be judged."""
    return not (p.bottom_cut and p.h < cfg.vest_min_visible_aspect * p.w)


def geometry_score(d, p):
    x1, y1, x2, y2 = p.xyxy
    w, h = max(p.w, 1.0), max(p.h, 1.0)
    region = np.array([x1 - cfg.x_margin * w, y1 - cfg.top_margin * h,
                       x2 + cfg.x_margin * w, y2 + 0.02 * h])
    ioa = _ioa(d.xyxy, region)
    if ioa < cfg.min_ioa:
        return 0.0, "outside_person"

    dx1, dy1, dx2, dy2 = d.xyxy
    dw, dh = dx2 - dx1, dy2 - dy1
    ry = ((dy1 + dy2) / 2 - y1) / h
    rx = ((dx1 + dx2) / 2 - x1) / w

    lo, hi, ideal = slot_band(p, d.slot)
    if not (lo <= ry <= hi):
        return 0.0, "not_on_head" if d.slot == "helmet" else "not_on_torso"
    if d.slot == "helmet":
        # A bent-over / half-body / occluded person has a short box, so the helmet looks large
        # relative to it; the helmet height cap is therefore computed from the "reference
        # height". For a standing person (h >= 2w) nothing changes at all.
        h_ref = max(h, cfg.full_body_aspect * w)
        if dw > cfg.helmet_max_w_frac * w or dh > cfg.helmet_max_h_frac * h_ref:
            return 0.0, "bad_size"
    elif dw < cfg.vest_min_w_frac * w or not (cfg.vest_min_h_frac * h <= dh <= cfg.vest_max_h_frac * h):
        return 0.0, "bad_size"

    span = max(ideal - lo, hi - ideal, 1e-6)
    fit_y = 1.0 - 0.5 * min(1.0, abs(ry - ideal) / span)
    fit_x = 1.0 - 0.5 * min(1.0, abs(rx - 0.5) / 0.6)
    return float(ioa * fit_y * fit_x), "ok"


def associate(persons, dets):
    for d in dets:
        d.owner, d.assoc, d.status, d.reason = -1, 0.0, "pending", ""
        best_i, best_s, reasons = -1, 0.0, []
        for i, p in enumerate(persons):
            s, why = geometry_score(d, p)
            if s > best_s:
                best_i, best_s = i, s
            elif s == 0.0:
                reasons.append(why)
        if best_i < 0:
            d.status = "rejected"
            d.reason = min(reasons, key=REASON_PRIORITY.index) if reasons else "no_person"
            continue
        d.owner, d.assoc = best_i, best_s
        if d.conf < cfg.min_conf[d.cls_name]:
            d.status, d.reason = "rejected", "low_conf"

    for i, p in enumerate(persons):
        p.frame = {}
        for slot in SLOTS:
            mine = [d for d in dets if d.owner == i and d.slot == slot and d.status == "pending"]
            pos = max((d for d in mine if d.positive), key=lambda d: d.evidence, default=None)
            neg = max((d for d in mine if not d.positive), key=lambda d: d.evidence, default=None)
            for d in mine:
                d.status, d.reason = "superseded", "duplicate"
            ep = pos.evidence if pos else 0.0
            en = neg.evidence if neg else 0.0

            if pos is None and neg is None:
                decision, ev, win = "unknown", 0.0, None
            elif pos is not None and neg is not None and abs(ep - en) < cfg.conflict_margin:
                pos.reason = neg.reason = "conflict"
                decision, ev, win = "unknown", 0.0, None
            elif ep >= en:
                decision, ev, win = "yes", ep, pos
            else:
                decision, ev, win = "no", en, neg

            if decision == "no" and ((slot == "helmet" and p.top_cut) or (slot == "vest" and not torso_visible(p))):
                win.reason = "not_judgeable"
                decision, ev, win = "unknown", 0.0, None

            if win is not None:
                win.status, win.reason = "accepted", ""
            p.frame[slot] = (decision, ev, win)
    return persons, dets


# ============================================================
# Section 6) Stage D — temporal memory + violation event log
# ============================================================
class SlotState:
    __slots__ = ("s", "state", "last_ev")

    def __init__(self):
        self.s, self.state, self.last_ev = 0.0, "unknown", -10**9


class TrackMemory:
    def __init__(self):
        self.tracks = {}

    def _absence_applies(self, p, slot):
        if cfg.absence_evidence <= 0 or p.h < cfg.absence_min_person_h:
            return False
        if (slot == "helmet" and p.top_cut) or (slot == "vest" and not torso_visible(p)):
            return False
        return p.frame[slot][0] == "unknown"

    def update(self, persons, frame_idx, evaluated):
        for p in persons:
            if p.track_id is None:
                for slot in SLOTS:
                    dec, ev, _ = p.frame.get(slot, ("unknown", 0.0, None))
                    p.status[slot], p.score[slot] = dec, ev
                continue

            t = self.tracks.get(p.track_id)
            if t is None:
                t = self.tracks[p.track_id] = {"helmet": SlotState(), "vest": SlotState(), "hits": 0,
                                               "last": frame_idx, "last_crop": -10**9, "hold": {}}
            t["last"] = frame_idx

            if evaluated:
                t["hits"] += 1
                for slot in SLOTS:
                    dec, ev, _ = p.frame[slot]
                    st = t[slot]
                    st.s *= cfg.decay
                    if dec == "yes":
                        st.s += ev
                        st.last_ev = frame_idx
                    elif dec == "no":
                        st.s -= ev
                        st.last_ev = frame_idx
                    elif self._absence_applies(p, slot):
                        st.s -= cfg.absence_evidence
                    st.s = float(np.clip(st.s, -cfg.s_max, cfg.s_max))

                    if st.s >= cfg.on_ok:
                        st.state = "yes"
                    elif st.s <= -cfg.on_violation and t["hits"] >= cfg.min_track_hits:
                        st.state = "no"
                    elif frame_idx - st.last_ev > cfg.stale_frames:
                        st.state = "unknown"

            for slot in SLOTS:
                p.status[slot], p.score[slot] = t[slot].state, t[slot].s

        self.tracks = {k: v for k, v in self.tracks.items() if frame_idx - v["last"] <= cfg.forget_frames}


class EventLog:
    """One event = a continuous interval during which a slot's state was "no" for one track ID."""

    def __init__(self, fps):
        self.fps, self.open, self.events = fps, {}, []

    def update(self, frame_idx, persons):
        confirmed = []
        visible = {p.track_id: p for p in persons if p.track_id is not None}
        for tid, p in visible.items():
            for slot in cfg.required_ppe:
                key = (tid, slot)
                if p.status.get(slot) == "no":
                    ev = self.open.get(key)
                    if ev is None:
                        ev = self.open[key] = {"track_id": tid, "ppe": slot, "start_frame": frame_idx,
                                               "last_frame": frame_idx, "confirmed": False}
                    ev["last_frame"] = frame_idx
                    if not ev["confirmed"] and frame_idx - ev["start_frame"] + 1 >= cfg.event_min_frames:
                        ev["confirmed"] = True
                        confirmed.append((p, slot))
                elif key in self.open:
                    self._close(key)
        for key in list(self.open):
            if key[0] not in visible and frame_idx - self.open[key]["last_frame"] > cfg.stale_frames:
                self._close(key)
        return confirmed

    def _close(self, key):
        ev = self.open.pop(key)
        if ev["confirmed"]:
            ev["start_s"] = round(ev["start_frame"] / self.fps, 2)
            ev["end_s"] = round(ev["last_frame"] / self.fps, 2)
            ev["duration_s"] = round((ev["last_frame"] - ev["start_frame"] + 1) / self.fps, 2)
            self.events.append(ev)

    def close_all(self):
        for key in list(self.open):
            self._close(key)

    def dataframe(self):
        cols = ["track_id", "ppe", "start_s", "end_s", "duration_s", "start_frame", "last_frame"]
        return pd.DataFrame(self.events, columns=cols) if self.events else pd.DataFrame(columns=cols)


# ============================================================
# Section 7) PPEPipeline
# ============================================================
@dataclass
class DrawItem:
    xyxy: np.ndarray
    cls_name: str
    conf: float
    held: bool = False
    owner: object = None        # the owning Person — for label layout


@dataclass
class FrameOutput:
    persons: list
    dets: list
    shown: list      # people that get drawn
    items: list      # helmet/vest DrawItems
    evaluated: bool


def _to_rel(box, p):
    x1, y1 = p.xyxy[:2]
    w, h = max(p.w, 1.0), max(p.h, 1.0)
    return np.array([(box[0] - x1) / w, (box[1] - y1) / h, (box[2] - x1) / w, (box[3] - y1) / h], np.float32)


def _from_rel(rel, p):
    x1, y1 = p.xyxy[:2]
    return np.array([x1 + rel[0] * p.w, y1 + rel[1] * p.h, x1 + rel[2] * p.w, y1 + rel[3] * p.h], np.float32)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (_area(a) + _area(b) - inter + 1e-6)


def display_box(win, dets):
    """
    (a) The crop is only for "scoring". If the winner came from a crop and the same class was
    also seen (overlapping) in the full-frame pass, the larger full-frame box is drawn — the
    same box the base model would draw.
    """
    if not cfg.display_full_box or win.source != "crop":
        return win.xyxy
    best, best_iou = None, cfg.display_match_iou
    for d in dets:
        if d.source == "full" and d.cls_name == win.cls_name:
            iou = _iou(d.xyxy, win.xyxy)
            if iou >= best_iou:
                best, best_iou = d, iou
    if best is not None and _area(best.xyxy) >= _area(win.xyxy):
        return best.xyxy
    return win.xyxy


class PPEPipeline:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tracker_model = YOLO(PERSON_WEIGHTS, task="detect")   # fresh tracker per video (same engine)
        self.memory = TrackMemory()
        self.frame_idx = 0
        self.timing = defaultdict(list)
        self.counters = Counter()
        self.seen_ids = set()

    # ---------- choose people for cropping ----------
    def _select_crops(self, persons, video_mode):
        if not cfg.crop_refine:
            return []
        wanted = {"unknown", "no"} if cfg.crop_verify_violations else {"unknown"}
        cands = []
        for i, p in enumerate(persons):
            if not (cfg.crop_min_person_h <= p.h <= cfg.crop_max_person_h):
                continue
            need = [s for s in cfg.required_ppe if p.frame[s][0] in wanted
                    and not (s == "helmet" and p.top_cut) and not (s == "vest" and not torso_visible(p))]
            if not need:
                continue
            t = self.memory.tracks.get(p.track_id) if (video_mode and p.track_id is not None) else None
            if t is not None:
                if self.frame_idx - t["last_crop"] < cfg.crop_cooldown_frames:
                    continue
                if all(abs(t[s].s) >= cfg.crop_skip_confident for s in need):
                    continue
            cands.append((p.conf, i))
        cands.sort(reverse=True)
        chosen = [i for _, i in cands[:cfg.crop_max_per_frame]]
        for i in chosen:
            t = self.memory.tracks.get(persons[i].track_id)
            if t is not None:
                t["last_crop"] = self.frame_idx
        return chosen

    # ---------- what gets drawn ----------
    def _display(self, persons, dets, video_mode):
        shown, items = [], []
        for p in persons:
            own = []
            t = self.memory.tracks.get(p.track_id) if (video_mode and p.track_id is not None) else None
            for slot in SLOTS:
                win = p.frame.get(slot, (None, None, None))[2]
                box = display_box(win, dets) if win is not None else None
                if t is None:
                    if win is not None:
                        own.append(DrawItem(box, win.cls_name, win.conf, owner=p))
                    continue
                state = p.status.get(slot)
                if state not in ("yes", "no"):
                    continue
                want_pos = state == "yes"
                if win is not None and win.positive == want_pos:
                    own.append(DrawItem(box, win.cls_name, win.conf, owner=p))
                    t["hold"][slot] = (win.cls_name, win.conf, _to_rel(box, p), self.frame_idx)
                else:
                    held = t["hold"].get(slot)
                    if held and SLOT_OF[held[0]][1] == want_pos and self.frame_idx - held[3] <= cfg.hold_frames:
                        own.append(DrawItem(_from_rel(held[2], p), held[0], held[1], held=True, owner=p))
            visible = (p.track_id is not None) if video_mode else (p.conf >= cfg.person_display_conf or bool(own))
            if visible:
                shown.append(p)
                items += own
        return shown, items

    def _tick(self, name, t0, model_ms=None):
        self.timing[name].append(time.perf_counter() - t0)
        if model_ms is not None:
            self.timing[name + "_model"].append(model_ms / 1000)

    def process(self, frame, video_mode=False):
        t = time.perf_counter()
        persons, ms = detect_persons(frame, self.tracker_model if video_mode else None)
        self._tick("A_person", t, ms)

        evaluated = (not video_mode) or (self.frame_idx % max(1, cfg.ppe_every_n_frames) == 0)
        dets = []
        if evaluated and (persons or not cfg.skip_ppe_without_person):
            t = time.perf_counter()
            dets, ms = detect_ppe_full(frame)
            self._tick("B1_ppe_full", t, ms)

            t = time.perf_counter()
            associate(persons, dets)
            self._tick("C_assoc", t)

            idxs = self._select_crops(persons, video_mode)
            if idxs:
                t = time.perf_counter()
                dets += detect_ppe_crops(frame, persons, idxs)
                associate(persons, dets)
                self._tick("B2_ppe_crop", t)
                self.counters["crops"] += len(idxs)
        else:
            for p in persons:
                p.frame = {s: ("unknown", 0.0, None) for s in SLOTS}

        t = time.perf_counter()
        if video_mode:
            self.memory.update(persons, self.frame_idx, evaluated)
        else:
            for p in persons:
                for s in SLOTS:
                    p.status[s], p.score[s] = p.frame[s][0], p.frame[s][1]
        shown, items = self._display(persons, dets, video_mode)
        self._tick("D_temporal+display", t)

        if cfg.show_skeleton:
            t = time.perf_counter()
            ms = attach_skeletons(frame, shown)
            self._tick("S_skeleton", t, ms)

        self.counters["frames"] += 1
        self.counters["ppe_raw"] += len(dets)
        for d in dets:
            self.counters[f"{d.status}:{d.reason}" if d.reason else d.status] += 1
        self.seen_ids.update(p.track_id for p in persons if p.track_id is not None)
        self.frame_idx += 1
        return FrameOutput(persons, dets, shown, items, evaluated)

    def timing_report(self):
        n = max(1, self.counters["frames"])
        rows = [{"stage": k, "ms_per_frame (avg over all frames)": round(1000 * sum(v) / n, 2),
                 "ms_per_call": round(1000 * sum(v) / max(1, len(v)), 2), "calls": len(v)}
                for k, v in sorted(self.timing.items())]
        total = sum(1000 * sum(v) / n for k, v in self.timing.items() if not k.endswith("_model"))
        rows.append({"stage": "TOTAL", "ms_per_frame (avg over all frames)": round(total, 2),
                     "ms_per_call": None, "calls": n})
        return pd.DataFrame(rows)

    def detection_report(self):
        c = {k: v for k, v in self.counters.items() if ":" in k or k in ("accepted", "ppe_raw", "crops")}
        return pd.Series(c, dtype=int).sort_values(ascending=False)


# ============================================================
# Section 8) HUD — the lightweight style of version 1
# ============================================================
STYLE = {   # BGR
    "Hardhat":        {"color": (60, 200, 60),  "thickness": 3, "friendly": "Helmet"},
    "NO-Hardhat":     {"color": (50, 50, 230),  "thickness": 3, "friendly": "No Helmet"},
    "Safety Vest":    {"color": (30, 210, 190), "thickness": 2, "friendly": "Vest"},
    "NO-Safety Vest": {"color": (0, 140, 255), "thickness": 2, "friendly": "No Vest"},
}
PERSON_STYLE = {"color": (230, 130, 40), "thickness": 1, "friendly": "Person"}
REJECT_COLOR = (170, 170, 170)
JUDGE_COLOR = (150, 200, 220)     # "not judgeable" — creamy gray
SKELETON_LINE = (235, 160, 70)     # light blue — fainter than the person frame
SKELETON_DOT = (255, 200, 110)
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
            (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]

BAR_HEIGHT = 42
ACCENT_HEIGHT = 5
UI_ALPHA = 0.65
TINT_ALPHA = 0.28
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_capsule(img, x1, y1, x2, y2, color, thickness=-1):
    radius = max(1, (y2 - y1) // 2)
    x1r, x2r = x1 + radius, max(x1 + radius, x2 - radius)
    cv2.rectangle(img, (x1r, y1), (x2r, y2), color, thickness, cv2.LINE_AA)
    cv2.circle(img, (x1r, y1 + radius), radius, color, thickness, cv2.LINE_AA)
    cv2.circle(img, (x2r, y1 + radius), radius, color, thickness, cv2.LINE_AA)


def corner_seg_length(x1, y1, x2, y2):
    return int(np.clip(0.35 * min(x2 - x1, y2 - y1), 12, 55))


def adaptive_thickness(x1, y1, x2, y2, base):
    return max(base, int(0.02 * min(x2 - x1, y2 - y1)))


def draw_corner_box(canvas, x1, y1, x2, y2, color, thickness, seg):
    for (cx, cy, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(canvas, (cx, cy), (cx + dx * seg, cy), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy), (cx, cy + dy * seg), color, thickness, cv2.LINE_AA)


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_len=6, gap_len=5):
    x1, y1 = pt1
    x2, y2 = pt2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist < 1:
        return
    step = dash_len + gap_len
    for i in range(int(dist // step) + 1):
        t0 = (i * step) / dist
        if t0 >= 1.0:
            break
        t1 = min(1.0, (i * step + dash_len) / dist)
        cv2.line(img, (int(x1 + (x2 - x1) * t0), int(y1 + (y2 - y1) * t0)),
                 (int(x1 + (x2 - x1) * t1), int(y1 + (y2 - y1) * t1)), color, thickness, cv2.LINE_AA)


def draw_box_connectors(img, x1, y1, x2, y2, color, seg, thickness=1):
    if x2 - seg > x1 + seg:
        draw_dashed_line(img, (x1 + seg, y1), (x2 - seg, y1), color, thickness)
        draw_dashed_line(img, (x1 + seg, y2), (x2 - seg, y2), color, thickness)
    if y2 - seg > y1 + seg:
        draw_dashed_line(img, (x1, y1 + seg), (x1, y2 - seg), color, thickness)
        draw_dashed_line(img, (x2, y1 + seg), (x2, y2 - seg), color, thickness)


def draw_skeleton_lines(img, kpts):
    vis = kpts[:, 2] >= cfg.kpt_conf
    pts = kpts[:, :2].astype(int)
    for a, b in SKELETON:
        if vis[a] and vis[b]:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), SKELETON_LINE, 1, cv2.LINE_AA)


def draw_skeleton_dots(img, kpts, k):
    r = max(2, int(round(2 * k)))
    for x, y, c in kpts:
        if c >= cfg.kpt_conf:
            cv2.circle(img, (int(x), int(y)), r, SKELETON_DOT, -1, cv2.LINE_AA)


def annotate(frame, out, fps=None):
    canvas = frame.copy()
    H, W = canvas.shape[:2]
    fs = max(0.5, W / 2560)                   # the font does not shrink at high resolutions
    ft = 1 if fs < 0.9 else 2
    k = fs / 0.5
    bar_h = int(BAR_HEIGHT * k)
    top_limit = ACCENT_HEIGHT + bar_h

    # --- list of things to draw: people first, then helmets/vests (their labels come on top) ---
    (_, th), baseline = cv2.getTextSize("Ag", FONT, fs, ft)
    pill_h = th + baseline + int(10 * k)

    draws = []   # (box, color, base thickness, tint?, label, capsule bottom)
    for p in out.shown:
        tid = f" #{p.track_id}" if p.track_id is not None else ""
        # the person label goes above its helmet label so they never collide
        bottom = min([p.xyxy[1]] + [it.xyxy[1] - pill_h - 2 for it in out.items
                                    if it.owner is p and it.xyxy[1] - pill_h < p.xyxy[1] + 2])
        draws.append((p.xyxy, PERSON_STYLE["color"], PERSON_STYLE["thickness"], False,
                      f"{PERSON_STYLE['friendly']}{tid} {p.conf:.0%}", bottom))
    for it in out.items:
        st = STYLE[it.cls_name]
        draws.append((it.xyxy, st["color"], st["thickness"], True, f"{st['friendly']} {it.conf:.0%}", it.xyxy[1]))

    layout = []
    for xyxy, color, base_th, tint, label, bottom in draws:
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        seg = corner_seg_length(x1, y1, x2, y2)
        th_box = adaptive_thickness(x1, y1, x2, y2, base_th)
        (tw, _), _ = cv2.getTextSize(label, FONT, fs, ft)
        pill_w = tw + pill_h + int(10 * k)
        py1 = max(top_limit + pill_h, int(bottom))
        px1 = int(np.clip(x1, 0, max(0, W - pill_w)))
        layout.append(dict(box=(x1, y1, x2, y2), color=color, th=th_box, conn=max(1, th_box // 2),
                           tint=tint, seg=seg, label=label, pill=(px1, py1 - pill_h, px1 + pill_w, py1)))

    # person states for the top bar
    counts = Counter()
    for p in out.shown:
        for slot, (yes_name, no_name) in {"helmet": ("Helmet", "No Helmet"), "vest": ("Vest", "No Vest")}.items():
            st = p.status.get(slot)
            if st == "yes":
                counts[yes_name] += 1
            elif st == "no":
                counts[no_name] += 1
    has_violation = any(p.status.get(s) == "no" for p in out.shown for s in cfg.required_ppe)
    skeletons = [p.kpts for p in out.shown if cfg.show_skeleton and p.kpts is not None]

    # --- layer 1: helmet/vest tint + all dashed connectors ---
    tint = canvas.copy()
    for d in layout:
        x1, y1, x2, y2 = d["box"]
        if d["tint"]:
            cv2.rectangle(tint, (x1, y1), (x2, y2), d["color"], -1)
        draw_box_connectors(tint, x1, y1, x2, y2, d["color"], d["seg"], d["conn"])
    canvas = cv2.addWeighted(tint, TINT_ALPHA, canvas, 1 - TINT_ALPHA, 0)

    # --- layer 2: capsules + status bar ---
    ui = canvas.copy()
    for kp in skeletons:                   # semi-transparent skeleton lines
        draw_skeleton_lines(ui, kp)
    for d in layout:
        draw_capsule(ui, *d["pill"], (20, 20, 20))
    cv2.rectangle(ui, (0, ACCENT_HEIGHT), (W, top_limit), (18, 18, 18), -1)
    canvas = cv2.addWeighted(ui, UI_ALPHA, canvas, 1 - UI_ALPHA, 0)

    # --- layer 3: sharp elements ---
    cv2.rectangle(canvas, (0, 0), (W, ACCENT_HEIGHT), (60, 60, 235) if has_violation else (70, 200, 70), -1)
    for kp in skeletons:                   # small skeleton dots
        draw_skeleton_dots(canvas, kp, k)

    if cfg.show_not_judgeable:
        shown_ids = {id(p) for p in out.shown}
        for dd in out.dets:
            if dd.reason != "not_judgeable" or dd.owner < 0 or id(out.persons[dd.owner]) not in shown_ids:
                continue
            x1, y1, x2, y2 = dd.xyxy.astype(int)
            draw_box_connectors(canvas, x1, y1, x2, y2, JUDGE_COLOR, 0, 1)
            text = f"{'Helmet' if dd.slot == 'helmet' else 'Vest'}? (not judgeable {dd.conf:.0%})"
            (tw, _), _ = cv2.getTextSize(text, FONT, fs * 0.8, 1)
            cv2.putText(canvas, text, (int(np.clip(x1 + 4, 0, max(0, W - tw))), min(H - 6, y1 + int(18 * k))),
                        FONT, fs * 0.8, JUDGE_COLOR, 1, cv2.LINE_AA)

    if cfg.debug_show_rejected:
        for dd in out.dets:
            if dd.status != "rejected":
                continue
            x1, y1, x2, y2 = dd.xyxy.astype(int)
            draw_box_connectors(canvas, x1, y1, x2, y2, REJECT_COLOR, 0, 1)
            text = f"x {dd.cls_name} {dd.conf:.2f} [{dd.reason}]"
            (tw, _), _ = cv2.getTextSize(text, FONT, fs * 0.8, 1)
            cv2.putText(canvas, text, (int(np.clip(x1, 0, max(0, W - tw))), max(top_limit + 12, y1 - 4)),
                        FONT, fs * 0.8, REJECT_COLOR, 1, cv2.LINE_AA)
    if cfg.debug_show_regions:
        for p in out.shown:
            x1, y1, x2, _ = p.xyxy.astype(int)
            for slot, col in (("helmet", (255, 220, 0)), ("vest", (0, 230, 255))):
                lo, hi, _ = slot_band(p, slot)
                for rel in (lo, hi):
                    yy = int(y1 + rel * p.h)
                    draw_dashed_line(canvas, (x1, yy), (x2, yy), col, 1, 4, 4)

    for d in layout:
        x1, y1, x2, y2 = d["box"]
        draw_corner_box(canvas, x1, y1, x2, y2, d["color"], d["th"], d["seg"])
        px1, py1, px2, py2 = d["pill"]
        ph = py2 - py1
        cv2.circle(canvas, (px1 + ph // 2, py1 + ph // 2), max(3, int(4 * k)), d["color"], -1, cv2.LINE_AA)
        cv2.putText(canvas, d["label"], (px1 + ph + int(4 * k), py2 - int(8 * k)),
                    FONT, fs, (255, 255, 255), ft, cv2.LINE_AA)

    bar_y = ACCENT_HEIGHT + bar_h // 2 + int(5 * k)
    title = "PPE MONITOR"
    (title_w, _), _ = cv2.getTextSize(title, FONT, fs * 1.2, ft)
    cv2.putText(canvas, title, (int(14 * k), bar_y), FONT, fs * 1.2, (255, 255, 255), ft, cv2.LINE_AA)

    slots = [(f"Person: {len(out.shown)}", PERSON_STYLE["color"])]
    for cls_name in ["Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest"]:
        st = STYLE[cls_name]
        slots.append((f"{st['friendly']}: {counts[st['friendly']]}", st["color"]))
    fps_text = f"{fps:.1f} FPS" if (fps is not None and cfg.show_fps) else ""
    (fps_w, _), _ = cv2.getTextSize(fps_text or " ", FONT, fs, ft)
    start_x = int(14 * k) + title_w + int(26 * k)
    slot_w = max(int(88 * k), (W - start_x - fps_w - int(24 * k)) // len(slots))
    cursor_x = start_x
    for text, color in slots:
        cv2.circle(canvas, (cursor_x, bar_y - int(5 * k)), max(3, int(4 * k)), color, -1, cv2.LINE_AA)
        cv2.putText(canvas, text, (cursor_x + int(10 * k), bar_y), FONT, fs, (235, 235, 235), ft, cv2.LINE_AA)
        cursor_x += slot_w
    if fps_text:
        cv2.putText(canvas, fps_text, (W - fps_w - int(14 * k), bar_y), FONT, fs, (190, 190, 190), ft, cv2.LINE_AA)
    return canvas


def persons_table(out):
    rows = []
    for p in out.persons:
        row = {"id": p.track_id, "conf": round(p.conf, 2), "h_px": int(p.h), "shown": any(p is q for q in out.shown),
               "overall": p.overall(cfg.required_ppe)}
        for s in SLOTS:
            dec, ev, win = p.frame.get(s, ("unknown", 0.0, None))
            row[f"{s}_frame"] = dec
            row[f"{s}_evidence"] = round(ev, 2)
            row[f"{s}_by"] = f"{win.cls_name}@{win.source}" if win else "-"
            row[f"{s}_final"] = p.status.get(s)
        rows.append(row)
    return pd.DataFrame(rows)


def show(img_bgr, title="", size=(12, 8)):
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(fa(title))
    plt.show()


# ============================================================
# Section 9) v1 vs v2 comparison on sample images
# ============================================================
def load_as_array(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        img = np.array(Image.open(image_path).convert("RGB"))[:, :, ::-1].copy()
    return img


V1_CLASS_IDS = []   # populated in init()


def v1_plot(img):
    return ppe_crop_model.predict(img, imgsz=640, conf=0.25, classes=V1_CLASS_IDS, device=DEVICE, verbose=False)[0].plot()


FATE_COLOR = {       # BGR — colors of the "differences" panel
    "accepted":      (80, 200, 80),     # 🟢 accepted
    "rejected":      (170, 170, 170),   # ⚪ rejected (with a reason)
    "not_judgeable": (60, 200, 230),    # 🟡 not judgeable
    "superseded":    (230, 150, 60),    # 🔵 duplicate / conflict — another box won
    "missing":       (200, 60, 200),    # 🟣 the base model saw it, we did not at all (e.g. TensorRT difference)
}


def _fate(d):
    return "not_judgeable" if d.reason == "not_judgeable" else d.status


def diff_panel(img, out, raw):
    """(c) all our PPE detections (full-frame + crop) on the image, color = fate, label = reason."""
    canvas = img.copy()
    H, W = canvas.shape[:2]
    fs = max(0.45, W / 2800)
    for p in out.persons:
        x1, y1, x2, y2 = p.xyxy.astype(int)
        draw_box_connectors(canvas, x1, y1, x2, y2, PERSON_STYLE["color"], 0, 1)
    for d in out.dets:
        col = FATE_COLOR[_fate(d)]
        x1, y1, x2, y2 = d.xyxy.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2 if d.status == "accepted" else 1, cv2.LINE_AA)
        tag = f"{d.cls_name} {d.conf:.2f}@{d.source}" + (f" [{d.reason}]" if d.reason else "")
        cv2.putText(canvas, tag, (max(0, x1), max(12, y1 - 4)), FONT, fs, col, 1, cv2.LINE_AA)
    for m in _missing(out, raw):
        x1, y1, x2, y2 = m["xyxy"].astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), FATE_COLOR["missing"], 2, cv2.LINE_AA)
        cv2.putText(canvas, f"MISSING {m['cls']} {m['conf']:.2f}", (max(0, x1), max(12, y1 - 4)),
                    FONT, fs, FATE_COLOR["missing"], 1, cv2.LINE_AA)
    return canvas


def _raw_ppe(raw):
    b = raw.boxes
    if b is None:
        return []
    rows = []
    for box, c, k in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)):
        if raw.names[k] in SLOT_OF:
            rows.append({"xyxy": box, "conf": float(c), "cls": raw.names[k]})
    return rows


def _missing(out, raw):
    full = [d for d in out.dets if d.source == "full"]
    return [r for r in _raw_ppe(raw)
            if not any(d.cls_name == r["cls"] and _iou(d.xyxy, r["xyxy"]) >= 0.5 for d in full)]


def match_table(out, raw):
    """(c) each base-model PPE detection ↔ its fate in our model + how much the drawn box differs from the base box."""
    full = [d for d in out.dets if d.source == "full"]
    rows = []
    for r in _raw_ppe(raw):
        cand = [(d, _iou(d.xyxy, r["xyxy"])) for d in full if d.cls_name == r["cls"]]
        d, iou = max(cand, key=lambda x: x[1], default=(None, 0.0))
        row = {"v1_class": r["cls"], "v1_conf": round(r["conf"], 2)}
        if d is None or iou < 0.5:
            row.update({"ours": "MISSING", "reason": "not seen in our full-frame pass (backend / NMS difference)",
                        "owner": "-", "shown_box_iou": "-", "shown_box_area_%": "-"})
            rows.append(row)
            continue
        owner = out.persons[d.owner] if d.owner >= 0 else None
        slot_win = owner.frame.get(d.slot, (None, None, None))[2] if owner is not None else None
        shown = [it for it in out.items if it.owner is owner and SLOT_OF[it.cls_name][0] == d.slot] if owner else []
        via_crop = (slot_win is not None and slot_win is not d and slot_win.source == "crop"
                    and slot_win.cls_name == d.cls_name and d.status == "superseded")
        if slot_win is d:
            ours, reason = "accepted", ""
        elif via_crop:
            ours, reason = "accepted (via crop)", "the crop scored higher; this full-frame box is the one drawn"
        else:
            ours, reason = _fate(d), d.reason
        row.update({"ours": ours, "reason": reason, "owner": d.owner})
        if shown:
            sb = shown[0].xyxy
            row["shown_box_iou"] = round(_iou(sb, r["xyxy"]), 2)
            row["shown_box_area_%"] = round(100 * _area(sb) / max(_area(r["xyxy"]), 1), 0)
        else:
            row["shown_box_iou"] = row["shown_box_area_%"] = "-"
        rows.append(row)
    return pd.DataFrame(rows)


def compare_on_image(img, title="", debug=False):
    prev = cfg.debug_show_rejected
    cfg.debug_show_rejected = debug
    out = pipe.process(img, video_mode=False)
    ours = annotate(img, out)
    cfg.debug_show_rejected = prev
    raw = ppe_crop_model.predict(img, imgsz=640, conf=0.25, classes=V1_CLASS_IDS, device=DEVICE, verbose=False)[0]

    fig, ax = plt.subplots(1, 3, figsize=(27, 8))
    panels = [(raw.plot(), "Base model (raw Snehil)"), (ours, "Ours (v4)"),
              (diff_panel(img, out, raw), "Differences: green=accepted  gray=rejected  yellow=not-judgeable  blue=duplicate  purple=missing")]
    for a, (im, t) in zip(ax, panels):
        a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        a.set_title(fa(f"{t}   {title}"), fontsize=11)
        a.axis("off")
    plt.tight_layout()
    plt.show()

    print("Base model ↔ ours match  (shown_box_area_% = our drawn box area relative to the base box):")
    display(match_table(out, raw))
    print("Per-person verdicts:")
    display(persons_table(out))
    if debug and out.dets:
        display(pd.DataFrame([{"class": d.cls_name, "conf": round(d.conf, 2), "src": d.source,
                               "owner": d.owner, "assoc": round(d.assoc, 2),
                               "status": d.status, "reason": d.reason} for d in out.dets]))
    return out

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".jfif", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ============================================================
# Section 11) Video processing
# ============================================================
def process_video(input_path, output_name, max_seconds=None, save_event_snapshots=True):
    pipe.reset()
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_path = OUTPUT_DIR / f"{output_name}_raw.mp4"
    final_path = OUTPUT_DIR / f"{output_name}.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (W, H))

    events = EventLog(fps_in)
    ev_dir = EVENTS_DIR / output_name
    ev_dir.mkdir(parents=True, exist_ok=True)

    frame_idx, fps_ema, violation_frames = 0, None, 0
    t_start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (max_seconds is not None and frame_idx / fps_in > max_seconds):
            break

        t = time.perf_counter()
        out = pipe.process(frame, video_mode=True)
        inst = 1.0 / max(time.perf_counter() - t, 1e-6)
        fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst

        annotated = annotate(frame, out, fps=fps_ema)
        violation_frames += int(any(p.overall(cfg.required_ppe) == "violation" for p in out.shown))

        for p, slot in events.update(frame_idx, out.persons):
            if save_event_snapshots:
                x1, y1, x2, y2 = person_crop_window(p, H, W)
                cv2.imwrite(str(ev_dir / f"id{p.track_id}_{slot}_f{frame_idx}.jpg"), annotated[y1:y2, x1:x2])

        # ZONES/ALERTS (section 14): the hook point, e.g.:
        # alerts.update(frame_idx, out.persons, annotated); report.update(frame_idx, out.persons)

        writer.write(annotated)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"\r  frame {frame_idx}/{total}  |  {fps_ema:.1f} FPS  |  events: {len(events.events) + sum(e['confirmed'] for e in events.open.values())}",
                  end="", flush=True)

    cap.release()
    writer.release()
    events.close_all()
    elapsed = time.perf_counter() - t_start
    print()
    if frame_idx == 0:
        raise RuntimeError(f"No frame could be read from {input_path}.")

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path),
                    "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-f", "mp4", str(final_path)], check=True)
    raw_path.unlink(missing_ok=True)

    summary = {
        "backend": f"person={PERSON_BACKEND}, ppe={PPE_BACKEND}, skeleton={POSE_BACKEND}",
        "resolution": f"{W}x{H}",
        "frames": frame_idx,
        "input_fps": round(fps_in, 2),
        "end_to_end_fps (incl. decode/draw/save)": round(frame_idx / elapsed, 2),
        "unique_person_ids": len(pipe.seen_ids),
        "violation_frame_ratio": round(violation_frames / frame_idx, 3),
        "violation_events": len(events.events),
        "crops_run": pipe.counters["crops"],
        "output": str(final_path),
    }
    return final_path, summary, events.dataframe()


def report_video(out_path, summary, events_df):
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nPer-stage time (ms):")
    display(pipe.timing_report())
    print("PPE detection fates:")
    display(pipe.detection_report())
    print("Violation events:")
    display(events_df)
    display(Video(str(out_path), embed=True, width=960))


# ============================================================
# Section 12) PyTorch FP16 vs TensorRT benchmark
# ============================================================
def _boxes(r):
    b = r.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    return b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)


def _iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-6)


def _agreement(ref_results, test_results):
    matched, total, diffs = 0, 0, []
    for ra, rb in zip(ref_results, test_results):
        (ba, ca, ka), (bb, cb, kb) = _boxes(ra), _boxes(rb)
        total += len(ba)
        iou = _iou_matrix(ba, bb)
        used = set()
        for i in np.argsort(-ca):
            best_j, best = -1, 0.5
            for j in range(len(bb)):
                if j not in used and kb[j] == ka[i] and iou[i, j] >= best:
                    best_j, best = j, iou[i, j]
            if best_j >= 0:
                used.add(best_j)
                matched += 1
                diffs.append(abs(ca[i] - cb[best_j]))
    return 100.0 * matched / max(total, 1), float(np.mean(diffs)) if diffs else float("nan")


def _timed_predict(model, frames, kw):
    model.predict(frames[0], **kw)
    t0 = time.perf_counter()
    res = [model.predict(f, **kw)[0] for f in frames]
    return res, 1000 * (time.perf_counter() - t0) / len(frames)


def benchmark_backends(video_path, n_frames=150):
    if "tensorrt" not in (PPE_BACKEND, PERSON_BACKEND):
        print("TensorRT is not active (current backend is PyTorch) — a comparison benchmark is meaningless.")
        return None
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < n_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    jobs = [
        ("PPE (Snehil)", YOLO(str(BEST_PT)), ppe_model, PPE_BACKEND,
         dict(imgsz=cfg.ppe_imgsz, conf=cfg.ppe_raw_conf, iou=cfg.ppe_iou, classes=PPE_CLASS_IDS)),
        (f"Person ({cfg.person_model})", YOLO(cfg.person_model), person_model, PERSON_BACKEND,
         dict(imgsz=cfg.person_imgsz, conf=cfg.person_conf, iou=cfg.person_iou, classes=[0])),
    ]
    if pose_model is not None:
        jobs.append((f"Skeleton ({cfg.pose_model})", YOLO(cfg.pose_model, task="pose"), pose_model, POSE_BACKEND,
                     dict(imgsz=cfg.person_imgsz, conf=0.25, classes=[0])))
    rows = []
    for name, m_pt, m_fast, be, kw in jobs:
        base = dict(device=DEVICE, verbose=False, **kw)
        r_pt, ms_pt = _timed_predict(m_pt, frames, {**base, **run_kw("pytorch")})
        r_fast, ms_fast = _timed_predict(m_fast, frames, {**base, **run_kw(be)})
        agree, cdiff = _agreement(r_pt, r_fast)
        n_pt = sum(len(_boxes(r)[0]) for r in r_pt)
        n_fast = sum(len(_boxes(r)[0]) for r in r_fast)
        rows.append({"model": name, "backend": "pytorch FP16", "ms_per_frame": round(ms_pt, 2),
                     "fps": round(1000 / ms_pt, 1), "detections": n_pt, "agreement_%": 100.0, "mean_abs_conf_diff": 0.0})
        rows.append({"model": name, "backend": be, "ms_per_frame": round(ms_fast, 2),
                     "fps": round(1000 / ms_fast, 1), "detections": n_fast, "agreement_%": round(agree, 2),
                     "mean_abs_conf_diff": round(cdiff, 4), "speedup": f"×{ms_pt / ms_fast:.2f}"})
    df = pd.DataFrame(rows)
    display(df)
    return df


# ============================================================
# Section 13-1) Base model only — image or video
# ============================================================
def run_raw_model(path, conf=0.25, show=True):
    """Raw Snehil output (all classes, PyTorch). Image → jpg, video → mp4 in OUTPUT_DIR."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        img = load_as_array(path)
        r = ppe_crop_model.predict(img, imgsz=640, conf=conf, device=DEVICE, verbose=False)[0]
        out_path = OUTPUT_DIR / f"{path.stem}_raw_model.jpg"
        cv2.imwrite(str(out_path), r.plot())
        table = pd.DataFrame([{"class": r.names[int(k)], "conf": round(float(c), 2)}
                              for c, k in zip(r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy())])
        if show:
            show_img = globals()["show"]
            show_img(r.plot(), f"Base model only — {path.name}")
            display(table)
        return {"output": str(out_path), "detections": table}
    if ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_mp4 = OUTPUT_DIR / f"{path.stem}_raw_model_tmp.mp4"
        out_path = OUTPUT_DIR / f"{path.stem}_raw_model.mp4"
        writer = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        counts, n = Counter(), 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            r = ppe_crop_model.predict(frame, imgsz=640, conf=conf, device=DEVICE, verbose=False, **run_kw("pytorch"))[0]
            counts.update(r.names[int(k)] for k in r.boxes.cls.cpu().numpy())
            writer.write(r.plot())
            n += 1
        cap.release()
        writer.release()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_mp4), "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p", "-f", "mp4", str(out_path)], check=True)
        raw_mp4.unlink(missing_ok=True)
        if show:
            print(f"{n} frames — detections across the whole video:", dict(counts))
            if Video is not None:
                display(Video(str(out_path), embed=True, width=960))
        return {"output": str(out_path), "counts": dict(counts)}
    raise ValueError(f"Unsupported format: {ext}")


# ============================================================
# Bootstrap: weights & engines on Drive (once) → fast loading (every run)
# ============================================================
def _gpu_tag():
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0).replace(" ", "_").replace("/", "_")


def _setup_dirs(base_dir):
    global BASE_DIR, WORK_DIR, WEIGHTS_DIR, ENGINE_DIR, OUTPUT_DIR, EVENTS_DIR, INPUT_DIR
    BASE_DIR = WORK_DIR = Path(base_dir)
    WEIGHTS_DIR = BASE_DIR / "weights"
    ENGINE_DIR = BASE_DIR / "engines" / _gpu_tag()
    OUTPUT_DIR = BASE_DIR / "outputs"
    EVENTS_DIR = OUTPUT_DIR / "events"
    INPUT_DIR = BASE_DIR / "inputs"
    for p in [WEIGHTS_DIR, ENGINE_DIR, OUTPUT_DIR, EVENTS_DIR, INPUT_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def _setup_persian_font():
    font = BASE_DIR / "Vazirmatn.ttf"
    try:
        if not font.exists():
            urllib.request.urlretrieve(
                "https://github.com/google/fonts/raw/main/ofl/vazirmatn/Vazirmatn%5Bwght%5D.ttf", font)
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = [font_manager.FontProperties(fname=str(font)).get_name(), "DejaVu Sans"]
    except Exception:
        plt.rcParams["font.family"] = ["DejaVu Sans"]


def _ensure_snehil_best():
    dst = WEIGHTS_DIR / "snehil_best.pt"
    if dst.exists() and dst.stat().st_size > 1_000_000:
        return dst
    for url in SNEHIL_BEST_URLS:
        try:
            urllib.request.urlretrieve(url, dst)
            if dst.stat().st_size > 1_000_000:
                return dst
        except Exception:
            pass
    tmp = Path("/tmp/snehil_repo")
    if not tmp.exists():
        subprocess.run(["git", "clone", "--depth", "1", SNEHIL_REPO + ".git", str(tmp)], check=True)
    src = next(tmp.rglob("best.pt"))
    shutil.copy(src, dst)
    return dst


def _ensure_ultralytics_asset(name):
    dst = WEIGHTS_DIR / name
    if dst.exists():
        return dst
    try:                                          # prefer the copy shipped in this repository
        local = Path(name)
        if local.exists() and local.stat().st_size > 1_000_000:
            shutil.copy(local, dst)
            return dst
        urllib.request.urlretrieve(f"{REPO_RAW}/{name}", dst)
        if dst.stat().st_size > 1_000_000:
            return dst
    except Exception:
        pass
    YOLO(name)                                   # fallback: Ultralytics downloads it to the CWD
    shutil.copy(name, dst)
    return dst


def build_trt_engine(pt_path, imgsz, task, rebuild=False):
    """TensorRT FP16 engine (batch=1, fixed imgsz input) — not rebuilt if it already exists on Drive."""
    target = ENGINE_DIR / f"{Path(str(pt_path)).stem}_{imgsz}_fp16.engine"
    if target.exists() and not rebuild:
        return target
    print(f"⏳ Building TensorRT engine for {Path(str(pt_path)).name} (first time only, a few minutes)...")
    t0 = time.perf_counter()
    local_pt = Path("/tmp") / Path(str(pt_path)).name   # do the export on local disk, not on Drive
    shutil.copy(pt_path, local_pt)
    out = YOLO(str(local_pt), task=task).export(format="engine", imgsz=imgsz, device=0, batch=1,
                                                dynamic=False, verbose=False, **FP16_KW)
    shutil.copy(str(out), str(target))
    print(f"✅ {target.name} saved to Drive ({time.perf_counter() - t0:.0f}s)")
    return target


def load_backend(pt_path, imgsz, task):
    """(model, backend, weights_path) — TensorRT from Drive first, rebuild if that fails, PyTorch if that fails."""
    if cfg.backend == "tensorrt" and torch.cuda.is_available():
        for rebuild in (False, True):
            try:
                eng = build_trt_engine(pt_path, imgsz, task, rebuild=rebuild)
                m = YOLO(str(eng), task=task)
                m.predict(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz, device=DEVICE, verbose=False)
                return m, "tensorrt", str(eng)
            except Exception as e:
                if rebuild:
                    print(f"⚠️ TensorRT failed for {Path(str(pt_path)).name} → PyTorch. ({type(e).__name__}: {e})")
    return YOLO(str(pt_path), task=task), "pytorch", str(pt_path)


def run_kw(backend):
    return FP16_KW if (backend == "pytorch" and cfg.fp16) else {}


def init(base_dir=DEFAULT_BASE_DIR, **overrides):
    """
    Call once per session. Every Config parameter can be overridden here:
        pm.init(backend="pytorch", show_skeleton=False, person_conf=0.15)
    """
    global ppe_model, ppe_crop_model, person_model, pose_model, pipe, BEST_PT
    global PPE_BACKEND, PERSON_BACKEND, POSE_BACKEND, PERSON_WEIGHTS, PPE_NAME_TO_ID, PPE_CLASS_IDS, _READY
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"Unknown parameter: {k}")
        setattr(cfg, k, v)

    t0 = time.perf_counter()
    _setup_dirs(base_dir)
    _setup_persian_font()
    BEST_PT = _ensure_snehil_best()
    person_pt = _ensure_ultralytics_asset(cfg.person_model)

    ppe_model, PPE_BACKEND, _ = load_backend(BEST_PT, cfg.ppe_imgsz, "detect")
    ppe_crop_model = YOLO(str(BEST_PT))
    person_model, PERSON_BACKEND, PERSON_WEIGHTS = load_backend(person_pt, cfg.person_imgsz, "detect")
    pose_model, POSE_BACKEND = None, "off"
    if cfg.show_skeleton:
        pose_pt = _ensure_ultralytics_asset(cfg.pose_model)
        pose_model, POSE_BACKEND, _ = load_backend(pose_pt, cfg.person_imgsz, "pose")

    PPE_NAME_TO_ID = {n: i for i, n in ppe_model.names.items()}
    missing = [n for n in PPE_CLASS_NAMES if n not in PPE_NAME_TO_ID]
    assert not missing, f"Classes {missing} are missing from the Snehil model"
    PPE_CLASS_IDS = [PPE_NAME_TO_ID[n] for n in PPE_CLASS_NAMES]
    V1_CLASS_IDS[:] = PPE_CLASS_IDS + [PPE_NAME_TO_ID["Person"]]
    assert person_model.names[0] == "person"

    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(2):
        person_model.predict(dummy, imgsz=cfg.person_imgsz, device=DEVICE, verbose=False, **run_kw(PERSON_BACKEND))
        ppe_model.predict(dummy, imgsz=cfg.ppe_imgsz, device=DEVICE, verbose=False, **run_kw(PPE_BACKEND))
        ppe_crop_model.predict([dummy[:320, :200]] * 2, imgsz=cfg.crop_imgsz, device=DEVICE, verbose=False,
                               **run_kw("pytorch"))
    LOGGER.setLevel(logging.ERROR)

    pipe = PPEPipeline()
    _READY = True
    print(f"✅ ppe_monitor {__version__} ready ({time.perf_counter() - t0:.0f}s) — "
          f"person={PERSON_BACKEND} | ppe={PPE_BACKEND} | skeleton={POSE_BACKEND} | GPU={_gpu_tag()}")
    print(f"   Outputs: {OUTPUT_DIR}")


# ============================================================
# Simple API: just give a path
# ============================================================
def _resolve(path):
    p = Path(path)
    if p.exists():
        return p
    for cand in (INPUT_DIR / path, BASE_DIR / path):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"File not found: {path} (default input folder: {INPUT_DIR})")


def analyze(path, show=True, name=None, max_seconds=None, benchmark=False, debug=False):
    """
    Analyzes an image or video (type from the extension). Output is saved to Drive/ppe_monitor/outputs.
    Returns: a dict with the output path and stats (for images: the persons table; for videos: summary + events).
    """
    if not _READY:
        init()
    src = _resolve(path)
    ext = src.suffix.lower()
    name = name or f"{src.stem}_v4"
    prev = cfg.debug_show_rejected
    cfg.debug_show_rejected = debug
    try:
        if ext in IMAGE_EXTS:
            img = load_as_array(src)
            out = pipe.process(img, video_mode=False)
            annotated = annotate(img, out)
            out_path = OUTPUT_DIR / f"{name}.jpg"
            cv2.imwrite(str(out_path), annotated)
            table = persons_table(out)
            if show:
                show_img = globals()["show"]
                show_img(annotated, f"{src.name}")
                display(table)
            print("Saved:", out_path)
            return {"output": str(out_path), "persons": table}
        if ext in VIDEO_EXTS:
            out_path, summary, events_df = process_video(src, name, max_seconds=max_seconds)
            if show:
                size_mb = Path(out_path).stat().st_size / 1e6
                if size_mb > 60:                       # do not embed a large video in the notebook
                    print("Summary:", summary)
                    display(pipe.timing_report()); display(events_df)
                    print(f"The video is large ({size_mb:.0f} MB) and was not embedded — open it from Drive: {out_path}")
                else:
                    report_video(out_path, summary, events_df)
            if benchmark:
                benchmark_backends(src)
            return {"output": str(out_path), "summary": summary, "events": events_df}
        raise ValueError(f"Unsupported format: {ext}")
    finally:
        cfg.debug_show_rejected = prev


def analyze_raw(path, **kw):
    """Base model only (raw Snehil, all classes) — no processing at all."""
    if not _READY:
        init()
    return run_raw_model(_resolve(path), **kw)


def compare(path, debug=False):
    """Three-panel comparison base model / ours / differences + match table (images only)."""
    if not _READY:
        init()
    return compare_on_image(load_as_array(_resolve(path)), Path(str(path)).name, debug=debug)


# ============================================================
# Zone / alert / report infrastructure (commented — next phase)
# ============================================================


# ============================================================
# Section 14) ZONES + ALERTS + DAILY REPORT — this whole cell is a comment
# ============================================================
# import requests
#
# # ---------------------------------------------------------------
# # 1) Zones — pixel coordinates on that camera's image (set them with one sample frame)
# # ---------------------------------------------------------------
# @dataclass
# class Zone:
#     name: str
#     polygon: list                         # [(x, y), ...]
#     required_ppe: tuple = ("helmet", "vest")
#
#     def __post_init__(self):
#         self._poly = np.array(self.polygon, np.int32).reshape(-1, 1, 2)
#
#     def contains(self, p):
#         """Is the foot point (middle of the box's bottom edge) inside the polygon?"""
#         foot = (float((p.xyxy[0] + p.xyxy[2]) / 2), float(p.xyxy[3]))
#         return cv2.pointPolygonTest(self._poly, foot, False) >= 0
#
#     def draw(self, img, color=(0, 200, 255)):
#         overlay = img.copy()
#         cv2.fillPoly(overlay, [self._poly], color)
#         cv2.addWeighted(overlay, 0.12, img, 0.88, 0, dst=img)
#         cv2.polylines(img, [self._poly], True, color, 2, cv2.LINE_AA)
#         cv2.putText(img, self.name, tuple(self._poly[0, 0]), FONT, 0.6, color, 2, cv2.LINE_AA)
#
#
# CAMERA_ZONES = {
#     # "cam_01": [Zone("Scaffold-A", [(120, 300), (900, 280), (1000, 700), (80, 720)])],
#     # "cam_02": [Zone("Gate", [(0, 400), (640, 400), (640, 720), (0, 720)], required_ppe=("helmet",))],
# }
#
#
# def zone_of(p, zones):
#     return next((z for z in zones if z.contains(p)), None)
#
#
# def apply_zones(out, zones):
#     """Removes people outside all zones from shown and from violation judging."""
#     if not zones:
#         return out
#     keep = [p for p in out.shown if zone_of(p, zones) is not None]
#     out.shown = keep
#     out.items = [it for it in out.items if any(it.owner is p for p in keep)]
#     return out
#
#
# # ---------------------------------------------------------------
# # 2) Alerts — Telegram (or any other webhook) with a per-person cooldown
# # ---------------------------------------------------------------
# class AlertNotifier:
#     def __init__(self, camera="cam_01", cooldown_s=60):
#         self.token = os.environ.get("PPE_TELEGRAM_TOKEN")      # never put the token inside the notebook
#         self.chat_id = os.environ.get("PPE_TELEGRAM_CHAT_ID")
#         self.camera, self.cooldown_s, self.last_sent = camera, cooldown_s, {}
#
#     def send(self, image_bgr, caption):
#         if not (self.token and self.chat_id):
#             print("[alert-dry-run]", caption)
#             return
#         ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
#         requests.post(f"https://api.telegram.org/bot{self.token}/sendPhoto",
#                       data={"chat_id": self.chat_id, "caption": caption},
#                       files={"photo": ("event.jpg", buf.tobytes(), "image/jpeg")}, timeout=10)
#
#     def on_confirmed(self, p, slot, annotated, t_sec, zone_name="-"):
#         key = (p.track_id, slot)
#         if t_sec - self.last_sent.get(key, -1e9) < self.cooldown_s:
#             return
#         self.last_sent[key] = t_sec
#         H, W = annotated.shape[:2]
#         x1, y1, x2, y2 = person_crop_window(p, H, W)
#         self.send(annotated[y1:y2, x1:x2],
#                   f"⚠️ {self.camera} | {zone_name} | ID {p.track_id} | NO {slot.upper()} | t={t_sec:.1f}s")
#
#
# # ---------------------------------------------------------------
# # 3) Compliance report — safe person-seconds / total, per zone per minute
# # ---------------------------------------------------------------
# class ComplianceReport:
#     def __init__(self, fps, camera="cam_01"):
#         self.dt, self.camera = 1.0 / fps, camera
#         self.acc = defaultdict(lambda: {"person_s": 0.0, "ok_s": 0.0, "violation_s": 0.0})
#
#     def update(self, frame_idx, persons, zones):
#         minute = int(frame_idx * self.dt // 60)
#         for p in persons:
#             z = zone_of(p, zones) if zones else None
#             if zones and z is None:
#                 continue
#             a = self.acc[(minute, z.name if z else "ALL")]
#             a["person_s"] += self.dt
#             o = p.overall(z.required_ppe if z else cfg.required_ppe)
#             if o == "ok":
#                 a["ok_s"] += self.dt
#             elif o == "violation":
#                 a["violation_s"] += self.dt
#
#     def dataframe(self):
#         rows = [{"camera": self.camera, "minute": m, "zone": z, **v,
#                  "compliance_%": round(100 * v["ok_s"] / v["person_s"], 1) if v["person_s"] else None}
#                 for (m, z), v in sorted(self.acc.items())]
#         return pd.DataFrame(rows)
#
#     def save(self, path):
#         self.dataframe().to_csv(path, index=False)
#
#
# # ---------------------------------------------------------------
# # 4) Hook into process_video (section 11) — inside the loop, after pipe.process:
# # ---------------------------------------------------------------
# # zones = CAMERA_ZONES.get("cam_01", [])
# # alerts = AlertNotifier("cam_01"); report = ComplianceReport(fps_in, "cam_01")
# # ...
# # out = apply_zones(pipe.process(frame, video_mode=True), zones)
# # annotated = annotate(frame, out, fps=fps_ema); [z.draw(annotated) for z in zones]
# # for p, slot in events.update(frame_idx, out.shown):
# #     z = zone_of(p, zones)
# #     alerts.on_confirmed(p, slot, annotated, frame_idx / fps_in, z.name if z else "-")
# # report.update(frame_idx, out.shown, zones)
# # ... after the loop:  report.save(OUTPUT_DIR / f"{output_name}_compliance.csv")
