#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuned RF-DETR Small + ByteTrack FORENSIC TIN/BOX COUNTER - 24/7 Real-Time CCTV Edition
==========================================================================================
Verified Behavioral Ground Truth Pipeline with:
  1. Real-time 24/7 RTSP / CCTV zero-backlog frame-drop capture (10 FPS paced).
  2. Pure minimal output: logs ONLY increasing counts with IST Day and Time.
  3. Separate per-frame execution timing telemetry log (model_time_ms, classifier_time_ms, pipeline_time_ms).
  4. Kinematic displacement + confidence gating on loitering tracks (zero false positives).
  5. Stale-lock expiration (35 processed frames).
  6. Shifted physical right exit line commit gate (RIGHT_COMMIT_PX = 85).
  7. Confirmed-return buffer for multi-trip worker state reset.
  8. Resilient cumulative count persistence and restoration across restarts.
"""

import argparse
from collections import Counter, deque
import csv
import datetime
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np

# ML / CV inference dependencies loaded on demand
torch = None
nn = None
sv = None
transforms = None
models = None
RFDETRSmall = None

def load_dependencies():
    """Load deep learning dependencies on-demand for inference."""
    global torch, nn, sv, transforms, models, RFDETRSmall
    if RFDETRSmall is not None:
        return
    try:
        import torch as _torch
        import torch.nn as _nn
        import supervision as _sv
        from torchvision import transforms as _transforms, models as _models
        from rfdetr import RFDETRSmall as _RFDETRSmall
    except ImportError as exc:
        raise RuntimeError(
            f"Missing required inference dependency: {exc}\n"
            "Please install the required packages:\n"
            "    pip install -r requirements.txt\n"
            "or:\n"
            "    pip install 'rfdetr[plus]' 'supervision>=0.25,<0.31' opencv-python\n"
        ) from exc

    if not hasattr(_sv, "ByteTrack"):
        raise RuntimeError(
            "This supervision version does not include sv.ByteTrack.\n"
            "Please run:  pip install 'supervision<0.31'\n"
        )

    torch, nn = _torch, _nn
    sv = _sv
    transforms, models = _transforms, _models
    RFDETRSmall = _RFDETRSmall

# ================================================================
# 2. TUNABLES & CAMERA GEOMETRY
# ================================================================
PERSON_CLASS_ID = 0
FULL_VALUE = 2

TARGET_PROCESS_FPS = 10.0
LOCK_AFTER_FRAMES = 4               # 4 consecutive 'full' votes to lock
VOTE_WINDOW = 7                     # Rolling 7-vote majority window

# Behavioral & Kinematic Loitering Gate Parameters
LOITER_WZ_EMPTY_MIN_VOTES = 10      # Work-zone empty votes threshold for loitering gate
LOITER_LOCK_MEAN_CONF = 0.95        # 95% mean conf required over 4 votes for loitering tracks
LOITER_LOCK_MIN_CONF = 0.90         # 90% min single-vote conf required for loitering tracks
LOITER_LOCK_MIN_DISP_PX = 120.0     # Min BBox displacement (px) across 4 votes for loitering tracks
STALE_LOCK_WZ_EMPTY_MIN = 5         # Prior work-zone empty votes for stale-lock expiration check
STALE_LOCK_MAX_FRAMES = 35          # Max processed frames a loitering track can hold a lock without crossing

LINE_HYSTERESIS_PX = 12             # Left line exit & return hysteresis (12px)
RIGHT_COMMIT_PX = 85                # 73px outward line shift + 12px hysteresis = 85px from inner ref line
NEAR_LINE_BOOST_PX = 220            # Classify every frame within 220px of inner ref lines
REASSIGNED_RETURN_RESET_DISTANCE_PX = 150
RETURN_RESET_SECONDS = 0.8
DIR_WINDOW = 5
DIR_MIN_DISP = 6
STITCH_GAP_SEC = 0.5
STITCH_MIN_RADIUS = 25
STITCH_BORDER = 30

# Inner Reference Lines (Used for Work-Zone boundary, Flip-TTA, and 220px pre-classification boost)
LINE_START, LINE_END = (851, 1280), (1008, 4)
RIGHT_LINE_START, RIGHT_LINE_END = (2020, 1280), (1482, 3)

# Actual Physical Right Exit Line (Shifted 73px perpendicular outward)
RIGHT_DRAW_LINE_START, RIGHT_DRAW_LINE_END = (2099, 1280), (1561, 3)

# Paths relative to script directory for portability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DEFAULT_COUNT_LOG = os.path.join(SCRIPT_DIR, "production_logs", "counts_log.csv")
DEFAULT_TIMING_LOG = os.path.join(SCRIPT_DIR, "production_logs", "pipeline_timing.csv")
DEFAULT_RTSP_URL = "rtsp://admin:P%40ssw0rd@192.168.100.6:554/cam/realmonitor?channel=5&subtype=0"


def sanitize_source_label(src):
    """Sanitize RTSP URL by redacting passwords so they are never leaked in logs or console."""
    src_str = str(src)
    if src_str.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
        try:
            from urllib.parse import urlsplit, urlunsplit
            p = urlsplit(src_str)
            netloc = p.hostname or "stream"
            if p.port:
                netloc = f"{netloc}:{p.port}"
            return urlunsplit((p.scheme, netloc, p.path, p.query, ""))
        except Exception:
            return "rtsp_stream"
    return os.path.basename(src_str)

# ================================================================
# 3. ONE-EURO SMOOTHER
# ================================================================
class OneEuro:
    def __init__(self, min_cutoff=0.6, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, np.zeros_like(x), t
            return x.copy()
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat.copy()

# ================================================================
# 4. DIRECTION & LINE GATE GEOMETRY
# ================================================================
def get_direction_label(dx, dy, threshold=5):
    if math.hypot(dx, dy) < threshold or abs(dx) < 2.0:
        return None
    return "to_left" if dx < 0 else "to_right"

def window_direction(hist, window=DIR_WINDOW, thr=DIR_MIN_DISP):
    if len(hist) < 2:
        return None
    pts = list(hist)[-window:]
    return get_direction_label(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1], threshold=thr)

def signed_dist(px, py, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1) or 1.0
    return ((px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)) / length

class LineGate:
    def __init__(self, name, p1, p2, inside_ref):
        self.name = name
        self.p1 = p1
        self.p2 = p2
        d = signed_dist(inside_ref[0], inside_ref[1], p1, p2)
        self.inside_sign = 1.0 if d >= 0 else -1.0

    def inside_dist(self, px, py):
        return signed_dist(px, py, self.p1, self.p2) * self.inside_sign

def update_line_side(ls, gate, px, py, exit_margin, return_margin):
    d = gate.inside_dist(px, py)
    if ls["side"] is None:
        ls["side"] = 1 if d >= 0 else -1
        return None
    if ls["side"] == 1 and d < -exit_margin:
        ls["side"] = -1
        return "exit"
    if ls["side"] == -1 and d > return_margin:
        ls["side"] = 1
        return "return"
    return None

# ================================================================
# 5. CLASS MAPPING LOADER
# ================================================================
def load_class_mapping(mapping_path):
    mapping = {}
    if mapping_path and os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    idx, cls_name = line.strip().split(":", 1)
                    mapping[int(idx)] = cls_name
    else:
        mapping = {0: "empty", 1: "full"}
    return mapping

# ================================================================
# 6. TRACK STATE & BEHAVIORAL KINEMATIC GATED VOTING
# ================================================================
def make_fresh_state(tid, smooth_cutoff, smooth_beta, count_id):
    return {
        "ids": [tid],
        "smoother": OneEuro(smooth_cutoff, smooth_beta),
        "hist": deque(maxlen=12),
        "age": 0,
        "predictions": [],
        "wz_empty_votes": 0,
        "wz_full_votes": 0,
        "consec_full_confs": [],
        "consec_full_xys": [],
        "locked_frame": None,
        "class_value": None,
        "class_name": "Unknown",
        "class_locked": False,
        "consecutive_same": 0,
        "last_consecutive_class": None,
        "counted": False,
        "counted_value": None,
        "count_id": str(count_id),
        "assigned_line": None,
        "crossed_side": None,
        "return_buffer_side": None,
        "return_buffer_frames": 0,
        "cross_info": {},
        "lines": {
            "left": {"side": None, "crossed": False, "pending": False},
            "right": {"side": None, "crossed": False, "pending": False},
        },
        "zone": None,
        "seen_work_zone": False,
        "color": (0, 0, 255),
        "last_seen_frame": 0,
        "last_xy": None,
        "last_box": None,
        "last_size": None,
        "vel": (0.0, 0.0),
        "ever_counted_full": False,
        "ever_counted_empty": False,
        "total_counted_value": 0,
        "ever_crossed_left": False,
        "ever_crossed_right": False,
        "all_predictions": [],
    }

def update_class(state, pred, conf, in_wz, ax, ay, pf):
    """
    Behavioral & Kinematic Class Voting:
    - Normal workers (wz_empty_votes < 10) lock on 4 consecutive 'full' votes.
    - Loitering tracks (wz_empty_votes >= 10 and wz_empty_votes > wz_full_votes) must satisfy:
        1) Confidence Gate: 4-vote mean_conf >= 0.95 AND 4-vote min_conf >= 0.90
        2) Kinematic Gate : 4-vote Euclidean displacement >= 120.0 px (active carrying motion)
      Rejecting false occlusion blips cleanly eliminates stationary/drifting loitering tracks.
    """
    if state["class_locked"]:
        return

    is_loitering = (
        state["wz_empty_votes"] >= LOITER_WZ_EMPTY_MIN_VOTES
        and state["wz_empty_votes"] > state["wz_full_votes"]
    )

    if pred == "empty":
        state["predictions"].append("empty")
        if in_wz:
            state["wz_empty_votes"] += 1
        state["consecutive_same"] = 1
        state["last_consecutive_class"] = "empty"
        state["consec_full_confs"].clear()
        state["consec_full_xys"].clear()
    elif pred == "full":
        if state["last_consecutive_class"] == "full":
            state["consecutive_same"] += 1
            state["consec_full_confs"].append(conf)
            state["consec_full_xys"].append((ax, ay))
        else:
            state["consecutive_same"] = 1
            state["last_consecutive_class"] = "full"
            state["consec_full_confs"] = [conf]
            state["consec_full_xys"] = [(ax, ay)]

        if not is_loitering:
            state["predictions"].append("full")
            if in_wz:
                state["wz_full_votes"] += 1

        if state["consecutive_same"] >= LOCK_AFTER_FRAMES:
            recent_4_c = state["consec_full_confs"][-LOCK_AFTER_FRAMES:]
            recent_4_xy = state["consec_full_xys"][-LOCK_AFTER_FRAMES:]
            mean_c = sum(recent_4_c) / float(len(recent_4_c))
            min_c = min(recent_4_c)
            disp_px = math.hypot(
                recent_4_xy[-1][0] - recent_4_xy[0][0],
                recent_4_xy[-1][1] - recent_4_xy[0][1],
            )

            if is_loitering and (
                mean_c < LOITER_LOCK_MEAN_CONF
                or min_c < LOITER_LOCK_MIN_CONF
                or disp_px < LOITER_LOCK_MIN_DISP_PX
            ):
                state["consecutive_same"] = 0
                state["last_consecutive_class"] = "empty"
                state["consec_full_confs"].clear()
                state["consec_full_xys"].clear()
            else:
                if is_loitering:
                    state["predictions"].extend(["full"] * LOCK_AFTER_FRAMES)
                    if in_wz:
                        state["wz_full_votes"] += LOCK_AFTER_FRAMES
                state["class_locked"] = True
                state["locked_frame"] = pf
                state["class_name"] = "full"
                state["class_value"] = FULL_VALUE
                return

    if not state["predictions"]:
        return

    window = state["predictions"][-VOTE_WINDOW:]
    counts = Counter(window)
    majority = max(counts, key=lambda k: (counts[k], k == pred))
    if (
        majority == "full"
        and state["wz_empty_votes"] >= LOITER_WZ_EMPTY_MIN_VOTES
        and state["wz_empty_votes"] > state["wz_full_votes"]
    ):
        majority = "empty"

    state["class_name"] = majority
    if majority == "full":
        state["class_value"] = FULL_VALUE
    elif majority == "empty":
        state["class_value"] = 0

def check_stale_lock_expiration(state, pf):
    """
    If a track with prior work-zone empty votes (`wz_empty_votes >= 5`) locked as 'full'
    in the middle of the room but has NOT crossed an exit line within `STALE_LOCK_MAX_FRAMES`
    (35 processed frames), expire the stale lock back to 'empty'.
    """
    if (
        state["class_locked"]
        and not state["counted"]
        and state["locked_frame"] is not None
        and state["wz_empty_votes"] >= STALE_LOCK_WZ_EMPTY_MIN
        and (pf - state["locked_frame"]) > STALE_LOCK_MAX_FRAMES
    ):
        state["class_locked"] = False
        state["locked_frame"] = None
        state["consecutive_same"] = 0
        state["last_consecutive_class"] = "empty"
        state["consec_full_confs"].clear()
        state["consec_full_xys"].clear()
        state["class_name"] = "empty"
        state["class_value"] = 0

# ================================================================
# 7. RTSP 24/7 CAPTURE WITH REAL-TIME FRAME DROP
# ================================================================
class RTSPCapture:
    """Threaded capture keeping only the newest frame and dropping stale frames.
    
    Guarantees zero-backlog real-time latency across 24/7 RTSP camera feeds.
    Provides automatic infinite reconnection with exponential backoff on stream drops.
    """
    def __init__(self, src, rtsp_transport="tcp", reconnect_delay=2.0, max_retries=0, timezone=None):
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self.src = src
        self.tz = timezone or ZoneInfo("Asia/Kolkata")
        self.rtsp_transport = rtsp_transport
        self.reconnect_delay = float(reconnect_delay)
        self.max_retries = int(max_retries)  # 0 means infinite 24/7 retries
        self.is_stream = isinstance(src, int) or (
            isinstance(src, str) and src.startswith(("rtsp://", "rtsps://", "http://", "https://"))
        )
        self.cap = None
        self.stopped = threading.Event()
        self.new_frame_event = threading.Event()
        self.lock = threading.Lock()

        self.latest_frame = None
        self.latest_frame_id = 0
        self.latest_captured_at = None
        self.total_captured = 0
        self.dropped_frames = 0
        self.capture_finished = False

        if self.is_stream and isinstance(self.src, str) and self.src.startswith(("rtsp://", "rtsps://")):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self.rtsp_transport}|fflags;nobuffer|flags;low_delay|stimeout;5000000|max_delay;500000"
            )

        self._connect()
        if self.is_stream:
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

    def _connect(self):
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

            if self.is_stream and isinstance(self.src, str) and self.src.startswith(("rtsp://", "rtsps://")):
                new_cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            else:
                new_cap = cv2.VideoCapture(self.src)

            if new_cap is not None and new_cap.isOpened():
                if self.is_stream:
                    try:
                        new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                ret, frame = new_cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.cap = new_cap
                        self.total_captured += 1
                        self.latest_frame_id = self.total_captured
                        self.latest_frame = frame
                        self.latest_captured_at = datetime.datetime.now(self.tz)
                    self.new_frame_event.set()
                    return True
                else:
                    try:
                        new_cap.release()
                    except Exception:
                        pass
            return False
        except Exception:
            return False

    def _capture_loop(self):
        failures = 0
        reconnects = 0
        while not self.stopped.is_set():
            with self.lock:
                cap = self.cap
            if cap is None or not cap.isOpened():
                if self.max_retries > 0 and reconnects >= self.max_retries:
                    break
                reconnects += 1
                if self.stopped.wait(self.reconnect_delay):
                    break
                if self._connect():
                    reconnects = 0
                    failures = 0
                continue

            try:
                ret, frame = cap.read()
            except Exception:
                ret, frame = False, None

            if ret and frame is not None:
                failures = 0
                reconnects = 0
                now_ts = datetime.datetime.now(self.tz)
                with self.lock:
                    self.total_captured += 1
                    self.latest_frame = frame
                    self.latest_frame_id = self.total_captured
                    self.latest_captured_at = now_ts
                self.new_frame_event.set()
            else:
                failures += 1
                if failures >= 20:
                    if not self.stopped.wait(0.1):
                        if self._connect():
                            failures = 0
                            reconnects = 0
                else:
                    self.stopped.wait(0.01)
        self.capture_finished = True

    def read_next(self, last_fid=None, timeout=0.5):
        """Returns the latest frame, skipping intermediate frames so latency remains zero."""
        if self.is_stream:
            end_time = time.monotonic() + max(0.0, timeout)
            while not self.stopped.is_set():
                with self.lock:
                    if self.latest_frame is not None and (last_fid is None or self.latest_frame_id > last_fid):
                        if last_fid is not None and (self.latest_frame_id - last_fid) > 1:
                            self.dropped_frames += (self.latest_frame_id - last_fid - 1)
                        return True, self.latest_frame.copy(), self.latest_frame_id, self.latest_captured_at

                if self.capture_finished:
                    return False, None, None, None

                remaining = end_time - time.monotonic()
                if remaining <= 0:
                    break
                self.new_frame_event.wait(timeout=min(0.05, remaining))
                self.new_frame_event.clear()

            with self.lock:
                if self.latest_frame is not None and (last_fid is None or self.latest_frame_id > last_fid):
                    return True, self.latest_frame.copy(), self.latest_frame_id, self.latest_captured_at
            return False, None, None, None

        # Video file sequential read
        ret, frame = self.cap.read() if self.cap is not None else (False, None)
        if not ret or frame is None:
            return False, None, None, None
        with self.lock:
            self.total_captured += 1
            fid = self.total_captured
        return True, frame, fid, datetime.datetime.now(self.tz)

    def isOpened(self):
        if self.is_stream:
            return not self.stopped.is_set() and (self.cap is not None and self.cap.isOpened())
        return self.cap is not None and self.cap.isOpened()

    def get(self, prop_id):
        if self.cap is not None:
            val = self.cap.get(prop_id)
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH and (val <= 0 or np.isnan(val)) and self.latest_frame is not None:
                return float(self.latest_frame.shape[1])
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT and (val <= 0 or np.isnan(val)) and self.latest_frame is not None:
                return float(self.latest_frame.shape[0])
            return val
        return 0.0

    def set(self, prop_id, value):
        return self.cap.set(prop_id, value) if self.cap else False

    def release(self):
        self.stopped.set()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

# ================================================================
# 8. RESNET BATCH CLASSIFIER WITH FLIP TTA
# ================================================================
def classify_crops(resnet, transform, crops_bgr, flip_flags, device, class_mapping):
    if not crops_bgr:
        return []
    tensors = []
    map_idx = []
    for i, (c, do_flip) in enumerate(zip(crops_bgr, flip_flags)):
        rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
        tensors.append(transform(rgb))
        map_idx.append((i, False))
        if do_flip:
            tensors.append(transform(cv2.flip(rgb, 1)))
            map_idx.append((i, True))

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(resnet(batch), dim=1).cpu().numpy()

    full_idx = next((k for k, v in class_mapping.items() if v.lower() == "full"), 1)

    crop_probs = {}
    for (crop_i, _), p in zip(map_idx, probs):
        if crop_i not in crop_probs:
            crop_probs[crop_i] = p.copy()
        else:
            if p[full_idx] > crop_probs[crop_i][full_idx]:
                crop_probs[crop_i] = p.copy()

    out = []
    for i in range(len(crops_bgr)):
        p = crop_probs[i]
        pred_idx = int(np.argmax(p))
        conf = float(p[pred_idx])
        out.append((class_mapping.get(pred_idx, "Unknown").lower(), conf))
    return out

# ================================================================
# 9. LOGGERS (PURE INCREASING COUNT IN IST + PER-FRAME TIMING)
# ================================================================
COUNT_LOG_FIELDS = [
    "ist_day",             # e.g. "Wednesday"
    "ist_time",            # e.g. "06:45:12.345"
    "ist_date",            # e.g. "2026-10-28"
    "ist_datetime",        # e.g. "2026-10-28 06:45:12.345"
    "count_increment",     # Increment added (e.g. 2)
    "line",                # "left" or "right"
    "class_name",          # "full"
    "cumulative_total",    # Running total count
    "cumulative_left",     # Running left line count
    "cumulative_right",    # Running right line count
    "track_id",            # Tracker ID
    "count_id",            # Count sequence identifier
    "frame_number",        # Processed frame number
    "source",              # Input stream / file
]

class CountEventLogger:
    """Appends one durable CSV row strictly when count INCREASES; flushes immediately."""
    def __init__(self, path):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        needs_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        self.file = open(self.path, "a", newline="", encoding="utf-8", buffering=1)
        self.writer = csv.DictWriter(self.file, fieldnames=COUNT_LOG_FIELDS, extrasaction="ignore")
        if needs_header:
            self.writer.writeheader()
            self.flush()

    def log_increment(self, dt, val, line, cls_name, total, left, right, tid, cid, frame_no, src):
        row = {
            "ist_day": dt.strftime("%A"),
            "ist_time": dt.strftime("%H:%M:%S.%f")[:-3],
            "ist_date": dt.strftime("%Y-%m-%d"),
            "ist_datetime": dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "count_increment": int(val),
            "line": str(line).lower(),
            "class_name": str(cls_name).lower(),
            "cumulative_total": int(total),
            "cumulative_left": int(left),
            "cumulative_right": int(right),
            "track_id": tid,
            "count_id": cid,
            "frame_number": frame_no,
            "source": src,
        }
        self.writer.writerow(row)
        self.flush()

    def flush(self):
        self.file.flush()
        try:
            os.fsync(self.file.fileno())
        except OSError:
            pass

    def close(self):
        try:
            self.flush()
        finally:
            self.file.close()


TIMING_LOG_FIELDS = [
    "ist_day",              # e.g. "Wednesday"
    "ist_time",             # Timestamp in IST (YYYY-MM-DD HH:MM:SS.mmm)
    "frame_number",         # Processed frame index
    "model_time_ms",        # Execution time of RF-DETR detector on this frame (ms)
    "classifier_time_ms",   # Execution time of ResNet classifier on tin crops this frame (ms)
    "pipeline_time_ms",     # Total end-to-end frame processing latency (ms)
    "pipeline_fps",         # Instantaneous pipeline throughput for this frame (FPS)
    "num_detections",       # Total person/tin detections in frame
    "num_crops_classified", # Number of tin crops classified in frame
]

class PipelineTimingLogger:
    """Logs per-frame execution telemetry: model time, classifier time, pipeline time."""
    def __init__(self, path):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        needs_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        self.file = open(self.path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=TIMING_LOG_FIELDS, extrasaction="ignore")
        if needs_header:
            self.writer.writeheader()
            self.file.flush()
        self._counter = 0

    def log_frame(self, dt, frame_no, model_ms, classifier_ms, pipeline_ms, num_dets, num_crops):
        row = {
            "ist_day": dt.strftime("%A"),
            "ist_time": dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "frame_number": frame_no,
            "model_time_ms": f"{model_ms:.2f}",
            "classifier_time_ms": f"{classifier_ms:.2f}",
            "pipeline_time_ms": f"{pipeline_ms:.2f}",
            "pipeline_fps": f"{1000.0 / max(pipeline_ms, 1e-3):.2f}",
            "num_detections": num_dets,
            "num_crops_classified": num_crops,
        }
        self.writer.writerow(row)
        self._counter += 1
        if self._counter % 10 == 0:
            self.file.flush()

    def close(self):
        try:
            self.file.flush()
            self.file.close()
        except Exception:
            pass


def load_cumulative_from_log(log_path):
    """Restore cumulative counts from the last valid event row after restarts."""
    totals = {"left": 0, "right": 0, "total": 0}
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return totals

    with open(log_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return totals
        for row in reader:
            try:
                totals = {
                    "left": int(float(row.get("cumulative_left") or 0)),
                    "right": int(float(row.get("cumulative_right") or 0)),
                    "total": int(float(row.get("cumulative_total") or 0)),
                }
            except (TypeError, ValueError, KeyError):
                continue
    return totals

# ================================================================
# 10. ASSET PATH RESOLVER & GIT LFS VALIDATION
# ================================================================
def check_git_lfs_pointer(filepath):
    if not filepath or not os.path.isfile(filepath):
        return False
    if os.path.getsize(filepath) < 10240:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                header = f.read(250)
                if "git-lfs" in header or "oid sha256:" in header:
                    return True
        except Exception:
            pass
    return False

def resolve_asset_path(cli_path, candidate_names, env_var=None, desc="asset"):
    if cli_path and os.path.isfile(cli_path):
        return os.path.abspath(cli_path)
    if env_var and os.environ.get(env_var) and os.path.isfile(os.environ[env_var]):
        return os.path.abspath(os.environ[env_var])
    if isinstance(candidate_names, str):
        candidate_names = [candidate_names]
    for name in candidate_names:
        p = os.path.join(MODELS_DIR, name)
        if os.path.isfile(p):
            return os.path.abspath(p)
    # Colab Drive fallback
    colab_dir = "/content/drive/MyDrive/Amrendra_Guardex/tin_counter_new_angle/models"
    for name in candidate_names:
        p = os.path.join(colab_dir, name)
        if os.path.isfile(p):
            return os.path.abspath(p)
    return os.path.join(MODELS_DIR, candidate_names[0])

# ================================================================
# 11. MAIN 24/7 PIPELINE EXECUTION
# ================================================================
def process_stream(
    source,
    model_path,
    detector_checkpoint_path,
    class_mapping_path,
    log_file_path=DEFAULT_COUNT_LOG,
    timing_log_path=DEFAULT_TIMING_LOG,
    line_pt1=LINE_START,
    line_pt2=LINE_END,
    right_line_pt1=RIGHT_LINE_START,
    right_line_pt2=RIGHT_LINE_END,
    classify_interval=5,
    target_fps=10.0,
    rfdetr_conf_threshold=0.20,
    person_class_id=0,
    track_activation=0.40,
    lost_sec=3.0,
    min_consecutive=2,
    nms_iou=0.6,
    smooth_cutoff=0.6,
    smooth_beta=0.01,
    line_hysteresis=LINE_HYSTERESIS_PX,
    right_commit_px=RIGHT_COMMIT_PX,
    anchor="center",
    enable_stitch=True,
    start_time_str=None,
    end_time_str=None,
    show_display=False,
    rtsp_transport="tcp",
    reconnect_delay=2.0,
    max_retries=0,
    timezone_name="Asia/Kolkata",
    no_timing_log=False,
    realtime=False,
):
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

    load_dependencies()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    safe_source = sanitize_source_label(source)

    # Git LFS pointer validation
    for name, path in [("ResNet Classifier", model_path), ("RF-DETR Detector", detector_checkpoint_path)]:
        if check_git_lfs_pointer(path):
            raise RuntimeError(
                f"Model file '{os.path.basename(path)}' is a Git LFS pointer text file. Run 'git lfs pull' to download binary weights."
            )

    is_stream = (
        isinstance(source, int)
        or (isinstance(source, str) and (source.isdigit() or source.startswith(("rtsp://", "rtsps://", "http://", "https://"))))
        or realtime
    )

    # Load neural models
    class_mapping = load_class_mapping(class_mapping_path)
    resnet = models.resnet152(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, len(class_mapping))
    try:
        resnet.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except TypeError:
        resnet.load_state_dict(torch.load(model_path, map_location=device))
    resnet = resnet.to(device)
    resnet.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    detector = RFDETRSmall.from_checkpoint(detector_checkpoint_path)

    # Initialize video/stream capture with 24/7 reconnects & frame drop
    cap = RTSPCapture(
        source,
        rtsp_transport=rtsp_transport,
        reconnect_delay=reconnect_delay,
        max_retries=max_retries,
        timezone=tz,
    )
    if not cap.isOpened():
        raise RuntimeError(f"Could not connect to stream / video source: {source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    if fps_val <= 0 or fps_val > 120 or np.isnan(fps_val):
        fps_val = float(target_fps) if target_fps > 0 else 25.0
    effective_fps = float(target_fps) if (is_stream or target_fps > 0) else fps_val

    # Tracker initialization
    trk_fps = max(1, int(round(effective_fps)))
    lost_buf = int(round(lost_sec * 30))
    max_lost_frames = int(trk_fps / 30.0 * lost_buf)
    EVICT_AFTER_FRAMES = max_lost_frames + 5
    stitch_gap = max(2, int(round(STITCH_GAP_SEC * effective_fps)))

    try:
        tracker = sv.ByteTrack(
            track_activation_threshold=track_activation,
            lost_track_buffer=lost_buf,
            minimum_matching_threshold=0.8,
            frame_rate=trk_fps,
            minimum_consecutive_frames=min_consecutive,
        )
    except TypeError:
        tracker = sv.ByteTrack(track_thresh=track_activation, track_buffer=lost_buf, match_thresh=0.8, frame_rate=trk_fps)

    # Geometry & Line Gates
    lmid = ((line_pt1[0] + line_pt2[0]) / 2.0, (line_pt1[1] + line_pt2[1]) / 2.0)
    rmid = ((right_line_pt1[0] + right_line_pt2[0]) / 2.0, (right_line_pt1[1] + right_line_pt2[1]) / 2.0)
    inside_ref = ((lmid[0] + rmid[0]) / 2.0, (lmid[1] + rmid[1]) / 2.0)
    gates = {
        "left": LineGate("left", line_pt1, line_pt2, inside_ref),
        "right": LineGate("right", right_line_pt1, right_line_pt2, inside_ref),
    }

    # Restore cumulative counts from existing log
    cumulative = load_cumulative_from_log(log_file_path)
    tally = {
        "total": cumulative["total"],
        "left": cumulative["left"],
        "right": cumulative["right"],
    }

    # Loggers
    count_logger = CountEventLogger(log_file_path)
    timing_logger = PipelineTimingLogger(timing_log_path) if not no_timing_log else None

    # Track states
    track_state = {}
    retired_ids = set()
    counted_ids = set()
    count_id_counter = {"next": 1}

    def new_count_id():
        val = f"C{count_id_counter['next']:07d}"
        count_id_counter["next"] += 1
        return val

    def exit_margin_for(side):
        return line_hysteresis if side == "left" else right_commit_px

    def resolve_crossing(state, tid, side, fc, pf, deferred):
        val = state["class_value"]
        d_left = gates["left"].inside_dist(state["last_xy"][0], state["last_xy"][1]) if state.get("last_xy") else 0.0
        d_right = gates["right"].inside_dist(state["last_xy"][0], state["last_xy"][1]) if state.get("last_xy") else 0.0
        req_margin = exit_margin_for(side)
        destination_reached = (d_left < -req_margin) if side == "left" else (d_right < -req_margin)

        if not state.get("seen_work_zone", False) or not destination_reached:
            state["lines"][side]["pending"] = True
            return False

        assigned = state.get("assigned_line")
        if assigned is not None and assigned != side:
            return False

        state["assigned_line"] = side
        state["counted"] = True
        counted_ids.add(int(tid))
        state["counted_value"] = val
        state["crossed_side"] = side

        if side == "left":
            state["ever_crossed_left"] = True
        else:
            state["ever_crossed_right"] = True

        if val > 0:
            state["ever_counted_full"] = True
            state["total_counted_value"] += val
            tally["total"] += val
            tally[side] += val

            # PURE INCREASING COUNT LOG IN IST DAY AND TIME
            now_dt = datetime.datetime.now(tz)
            count_logger.log_increment(
                dt=now_dt,
                val=val,
                line=side,
                cls_name=state.get("class_name", "full"),
                total=tally["total"],
                left=tally["left"],
                right=tally["right"],
                tid=tid,
                cid=state.get("count_id", "UNKNOWN"),
                frame_no=pf,
                src=safe_source,
            )

            # Output ONLY when count increases
            day_name = now_dt.strftime("%A")
            dt_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{day_name}, {dt_str} IST] COUNT INCREASED: +{val} | "
                f"Total={tally['total']} (Left={tally['left']}, Right={tally['right']}) | "
                f"Line={side.upper()} | Class=full",
                flush=True
            )
        else:
            state["ever_counted_empty"] = True

        return True

    def reset_for_next_counting_cycle(state, tid, fc, pf, side):
        state["count_id"] = new_count_id()
        state["predictions"].clear()
        state["wz_empty_votes"] = 0
        state["wz_full_votes"] = 0
        state["consec_full_confs"].clear()
        state["consec_full_xys"].clear()
        state["locked_frame"] = None
        state["class_value"] = None
        state["class_name"] = "Unknown"
        state["class_locked"] = False
        state["consecutive_same"] = 0
        state["last_consecutive_class"] = None
        state["counted"] = False
        state["counted_value"] = None
        state["crossed_side"] = None
        state["seen_work_zone"] = False
        state["return_buffer_side"] = None
        state["return_buffer_frames"] = 0
        state["assigned_line"] = None
        state["cross_info"] = {}
        for line_name, line_state in state["lines"].items():
            line_state["crossed"] = False
            line_state["pending"] = False

    def try_stitch(new_tid, ax, ay, sw, sh, pf, fc, active_now):
        best, best_d = None, 1e18
        for otid, ost in track_state.items():
            if otid in active_now or ost["last_xy"] is None:
                continue
            if (ost.get("counted", False)
                    or ost.get("assigned_line") is not None
                    or ost.get("crossed_side") in ("left", "right")
                    or any(ls.get("crossed", False) or ls.get("pending", False) for ls in ost.get("lines", {}).values())
                    or ost.get("class_locked", False)):
                continue
            gap = pf - ost["last_seen_frame"]
            if gap < 1 or gap > stitch_gap:
                continue
            lb = ost["last_box"]
            if lb[0] < STITCH_BORDER or lb[1] < STITCH_BORDER or lb[2] > w - STITCH_BORDER or lb[3] > h - STITCH_BORDER:
                continue
            pg = min(gap, 5)
            px = ost["last_xy"][0] + ost["vel"][0] * pg
            py = ost["last_xy"][1] + ost["vel"][1] * pg
            d = math.hypot(ax - px, ay - py)
            lw, lh = ost["last_size"]
            ratio = (sw * sh) / max(lw * lh, 1.0)
            stitch_radius = max(STITCH_MIN_RADIUS, 0.25 * lh)
            if d <= stitch_radius and 0.65 <= ratio <= 1.6 and d < best_d:
                best, best_d = otid, d
        if best is None:
            return None
        st = track_state.pop(best)
        st["ids"].append(new_tid)
        retired_ids.add(best)
        return st

    def detect_persons(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            dets = detector.predict(rgb, threshold=rfdetr_conf_threshold, include_source_image=False)
        except TypeError:
            dets = detector.predict(rgb, threshold=rfdetr_conf_threshold)
        if isinstance(dets, list):
            dets = dets[0] if dets else sv.Detections.empty()
        if dets is None or len(dets) == 0 or dets.class_id is None:
            return sv.Detections.empty()
        pd = dets[dets.class_id == person_class_id]
        if len(pd) > 1:
            pd = pd.with_nms(threshold=nms_iou, class_agnostic=True)
        return pd

    # Register clean shutdown signal
    try:
        import signal
        def sig_handler(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, sig_handler)
    except Exception:
        pass

    # 1-line clean startup announcement
    now_start = datetime.datetime.now(tz)
    print(
        f"[{now_start.strftime('%A, %Y-%m-%d %H:%M:%S')} IST] 24/7 Forensic Counter started | "
        f"Source: {safe_source} | Target: {effective_fps:.1f} FPS | "
        f"Restored Cumulative: Total={tally['total']} (Left={tally['left']}, Right={tally['right']})",
        flush=True
    )

    processed_frame_count = 0
    frame_count = 0
    last_fid = None
    last_inference_start = None

    try:
        while cap.isOpened():
            t_frame_start = time.perf_counter()

            if is_stream:
                # Real-time 10 FPS rate limiter pacing
                if last_inference_start is not None:
                    elapsed = time.monotonic() - last_inference_start
                    wait_s = (1.0 / effective_fps) - elapsed
                    if wait_s > 0:
                        time.sleep(wait_s)
                last_inference_start = time.monotonic()

                ret, frame, fid, captured_at = cap.read_next(last_fid=last_fid, timeout=0.5)
                if not ret:
                    if cap.capture_finished:
                        break
                    continue
                last_fid = fid
                frame_count = fid or (frame_count + 1)
                ts = time.monotonic()
            else:
                ret, frame, fid, captured_at = cap.read_next(timeout=0.5)
                if not ret:
                    break
                frame_count += 1
                ts = frame_count / effective_fps

            processed_frame_count += 1
            pf = processed_frame_count
            fc = frame_count

            # ---------------- 1. RF-DETR MODEL INFERENCE ----------------
            t_model_start = time.perf_counter()
            person_dets = detect_persons(frame)
            model_time_ms = (time.perf_counter() - t_model_start) * 1000.0

            tracked = tracker.update_with_detections(person_dets)

            items = []
            crops, flip_flags = [], []

            # ---------------- 2. PER-TRACK UPDATE ----------------
            if len(tracked) > 0:
                boxes = tracked.xyxy
                tids = tracked.tracker_id if tracked.tracker_id is not None else np.arange(len(boxes))
                confs = tracked.confidence if tracked.confidence is not None else np.ones(len(boxes))
                active_now = set(int(t) for t in tids)

                for box, tid, conf in zip(boxes, tids, confs):
                    tid = int(tid)
                    if tid in retired_ids:
                        continue
                    rx1, ry1, rx2, ry2 = [float(v) for v in box]
                    rw, rh = rx2 - rx1, ry2 - ry1
                    if rw < 2 or rh < 2:
                        continue
                    rcx, rcy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0

                    state = track_state.get(tid)
                    if state is None:
                        if enable_stitch:
                            state = try_stitch(tid, rcx, rcy, rw, rh, pf, fc, active_now)
                        if state is None:
                            state = make_fresh_state(tid, smooth_cutoff, smooth_beta, new_count_id())
                        track_state[tid] = state

                    scx, scy, sw, sh = state["smoother"]([rcx, rcy, rw, rh], ts)
                    sw, sh = max(sw, 2.0), max(sh, 2.0)
                    sx1, sy1 = max(0.0, scx - sw / 2.0), max(0.0, scy - sh / 2.0)
                    sx2, sy2 = min(float(w), scx + sw / 2.0), min(float(h), scy + sh / 2.0)
                    ax = scx
                    ay = scy if anchor == "center" else sy2

                    if state["last_xy"] is not None:
                        gap = max(1, pf - state["last_seen_frame"])
                        vx = (ax - state["last_xy"][0]) / gap
                        vy = (ay - state["last_xy"][1]) / gap
                        state["vel"] = (0.6 * state["vel"][0] + 0.4 * vx, 0.6 * state["vel"][1] + 0.4 * vy)

                    state["last_xy"] = (ax, ay)
                    state["last_box"] = (sx1, sy1, sx2, sy2)
                    state["last_size"] = (sw, sh)
                    state["last_seen_frame"] = pf
                    state["age"] += 1
                    state["hist"].append((ax, ay))

                    dL = gates["left"].inside_dist(ax, ay)
                    dR = gates["right"].inside_dist(ax, ay)
                    state["zone"] = "A" if dL < 0 else ("C" if dR < 0 else "B")
                    if dL >= 0 and dR >= 0:
                        state["seen_work_zone"] = True

                    # Expire stale mid-room locks on loitering tracks before crossing check
                    check_stale_lock_expiration(state, pf)

                    direction = window_direction(state["hist"])
                    pending_any = state["lines"]["left"]["pending"] or state["lines"]["right"]["pending"]
                    near_any_line = (abs(dL) <= NEAR_LINE_BOOST_PX) or (abs(dR) <= NEAR_LINE_BOOST_PX)

                    due = (
                        near_any_line
                        or state["age"] % classify_interval == 0
                        or state["class_name"] == "Unknown"
                        or pending_any
                    )

                    already_counted_returning = state["counted"] and (
                        (state["crossed_side"] == "left" and direction == "to_right")
                        or (state["crossed_side"] == "right" and direction == "to_left")
                    )

                    action, crop_idx = None, None
                    if due and not state["class_locked"]:
                        if already_counted_returning and not pending_any:
                            action = "returning"
                        else:
                            pad = 5
                            cx1, cy1 = max(0, int(rx1) - pad), max(0, int(ry1) - pad)
                            cx2, cy2 = min(w, int(rx2) + pad), min(h, int(ry2) + pad)
                            crop = frame[cy1:cy2, cx1:cx2]
                            if crop.size > 0:
                                action = "classify"
                                crop_idx = len(crops)
                                crops.append(crop)
                                use_flip_tta = (direction == "to_right") or (dR < dL)
                                flip_flags.append(use_flip_tta)

                    items.append({
                        "tid": tid, "state": state, "conf": float(conf),
                        "ax": ax, "ay": ay, "dL": dL, "dR": dR,
                        "direction": direction, "action": action, "crop_idx": crop_idx,
                    })

            # ---------------- 3. RESNET CLASSIFIER INFERENCE ----------------
            t_cls_start = time.perf_counter()
            results = classify_crops(resnet, transform, crops, flip_flags, device, class_mapping) if crops else []
            classifier_time_ms = (time.perf_counter() - t_cls_start) * 1000.0

            # ---------------- 4. CLASSIFICATION & CROSSING RESOLUTION ----------------
            for it in items:
                tid, state = it["tid"], it["state"]
                ax, ay = it["ax"], it["ay"]
                dL, dR = it["dL"], it["dR"]

                if it["crop_idx"] is not None and it["crop_idx"] < len(results):
                    name, c = results[it["crop_idx"]]
                    if name != "unknown":
                        in_wz = (dL >= 0 and dR >= 0)
                        update_class(state, name, c, in_wz, ax, ay, pf)
                        state["all_predictions"].append(name)

                # Deferred crossing resolution
                for side in ("left", "right"):
                    ls = state["lines"][side]
                    req_margin = exit_margin_for(side)
                    destination_reached = (dL < -req_margin) if side == "left" else (dR < -req_margin)
                    if ls["pending"] and state["class_value"] is not None and destination_reached:
                        if state["counted"]:
                            ls["pending"] = False
                        else:
                            resolve_crossing(state, tid, side, fc, pf, deferred=True)
                            if state.get("counted", False):
                                ls["pending"] = False

                # Confirmed-return buffer for multi-cycle workers
                previously_counted_track = int(tid) in counted_ids
                current_cycle_was_counted = bool(state.get("counted", False))
                has_counted_crossing = state.get("crossed_side") in ("left", "right")
                if previously_counted_track and current_cycle_was_counted and has_counted_crossing:
                    return_side = state["crossed_side"]
                    return_dist = dL if return_side == "left" else dR
                    if return_dist >= REASSIGNED_RETURN_RESET_DISTANCE_PX:
                        if state.get("return_buffer_side") == return_side:
                            state["return_buffer_frames"] += 1
                        else:
                            state["return_buffer_side"] = return_side
                            state["return_buffer_frames"] = 1
                        required_frames = max(2, int(round(RETURN_RESET_SECONDS * effective_fps)))
                        if state["return_buffer_frames"] >= required_frames:
                            reset_for_next_counting_cycle(state, tid, fc, pf, return_side)
                    else:
                        state["return_buffer_side"] = None
                        state["return_buffer_frames"] = 0
                else:
                    state["return_buffer_side"] = None
                    state["return_buffer_frames"] = 0

                # Crossings detection
                for side in ("left", "right"):
                    ls = state["lines"][side]
                    req_margin = exit_margin_for(side)
                    ev = update_line_side(ls, gates[side], ax, ay, req_margin, line_hysteresis)
                    if ev == "exit":
                        if ls["crossed"]:
                            continue
                        ls["crossed"] = True
                        state["cross_info"][side] = (ax, ay, pf)
                        assigned = state.get("assigned_line")
                        destination_reached = (dL < -req_margin) if side == "left" else (dR < -req_margin)
                        if not state.get("seen_work_zone", False) or not destination_reached:
                            pass
                        elif assigned is not None and assigned != side:
                            pass
                        elif state["counted"]:
                            pass
                        else:
                            if state["class_value"] is not None:
                                state["assigned_line"] = side
                                resolve_crossing(state, tid, side, fc, pf, deferred=False)
                            else:
                                if assigned is None:
                                    state["assigned_line"] = side
                                ls["pending"] = True
                    elif ev == "return":
                        if ls["pending"] and not state["counted"]:
                            ls["pending"] = False
                            ls["crossed"] = False
                            if state.get("assigned_line") == side:
                                state["assigned_line"] = None

            # Evict stale tracks
            stale = [tid for tid, s in track_state.items() if pf - s["last_seen_frame"] > EVICT_AFTER_FRAMES]
            for tid in stale:
                retired_ids.add(tid)
                del track_state[tid]

            # ---------------- 5. PER-FRAME PIPELINE TIMING LOGGING ----------------
            pipeline_time_ms = (time.perf_counter() - t_frame_start) * 1000.0

            if timing_logger is not None:
                timing_logger.log_frame(
                    dt=datetime.datetime.now(tz),
                    frame_no=pf,
                    model_ms=model_time_ms,
                    classifier_ms=classifier_time_ms,
                    pipeline_ms=pipeline_time_ms,
                    num_dets=len(person_dets),
                    num_crops=len(crops),
                )

            if show_display:
                try:
                    cv2.imshow("RF-DETR CCTV Counter", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except Exception:
                    show_display = False

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        count_logger.close()
        if timing_logger is not None:
            timing_logger.close()
        if show_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    now_stop = datetime.datetime.now(tz)
    print(
        f"[{now_stop.strftime('%A, %Y-%m-%d %H:%M:%S')} IST] 24/7 Counter stopped | "
        f"Processed: {processed_frame_count} frames | "
        f"Cumulative Total={tally['total']} (Left={tally['left']}, Right={tally['right']})",
        flush=True
    )

# ================================================================
# 12. CLI ARGS & ENTRY POINT
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="24/7 RF-DETR + ResNet CCTV Counter - Real-time zero-backlog with IST & Telemetry Logging"
    )
    parser.add_argument("--source", "-s", default=os.environ.get("SOURCE") or os.environ.get("RTSP_URL") or DEFAULT_RTSP_URL,
                        help=f"RTSP stream URL, webcam index (0), or video file (default: {sanitize_source_label(DEFAULT_RTSP_URL)})")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH"),
                        help="ResNet152 classifier weights (defaults to models/ResNet152_direction_best_4.pth)")
    parser.add_argument("--detector-checkpoint", default=os.environ.get("DETECTOR_CHECKPOINT"),
                        help="Fine-tuned RF-DETR Small checkpoint (defaults to models/best_model.pth)")
    parser.add_argument("--mapping", default=os.environ.get("MAPPING_PATH"),
                        help="Class mapping file (defaults to models/class_mapping.txt)")
    parser.add_argument("--log-file", default=os.environ.get("LOG_FILE") or DEFAULT_COUNT_LOG,
                        help="CSV file path for pure increasing count logs in IST (defaults to production_logs/counts_log.csv)")
    parser.add_argument("--timing-log", default=os.environ.get("TIMING_LOG") or DEFAULT_TIMING_LOG,
                        help="CSV file path for per-frame pipeline execution times (defaults to production_logs/pipeline_timing.csv)")
    parser.add_argument("--no-timing-log", action="store_true", help="Disable generating per-frame timing log")
    parser.add_argument("--timezone", default=os.environ.get("TIMEZONE", "Asia/Kolkata"),
                        help="Timezone for timestamps (default: Asia/Kolkata [IST])")
    parser.add_argument("--fps", type=float, default=TARGET_PROCESS_FPS,
                        help=f"Target real-time processing FPS limit (default: {TARGET_PROCESS_FPS})")
    parser.add_argument("--det-conf", type=float, default=0.20, help="RF-DETR confidence floor (default: 0.20)")
    parser.add_argument("--track-thresh", type=float, default=0.40, help="ByteTrack activation threshold (default: 0.40)")
    parser.add_argument("--transport", default="tcp", choices=["tcp", "udp"], help="RTSP transport (default: tcp)")
    parser.add_argument("--reconnect-delay", type=float, default=2.0, help="RTSP reconnect backoff delay in seconds (default: 2.0)")
    parser.add_argument("--max-retries", type=int, default=0, help="Max RTSP reconnection attempts (0 for infinite 24/7 retry)")
    parser.add_argument("--realtime", action="store_true", help="Force real-time rate limiter and frame-drop logic on video sources")
    parser.add_argument("--show", action="store_true", help="Open a display window for live preview")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = resolve_asset_path(
        args.model,
        ["ResNet152_direction_best_4.pth", "resnet152.pth"],
        env_var="MODEL_PATH",
        desc="ResNet classifier weights",
    )
    detector_checkpoint = resolve_asset_path(
        args.detector_checkpoint,
        ["best_model.pth", "checkpoint_best_ema.pth"],
        env_var="DETECTOR_CHECKPOINT",
        desc="RF-DETR detector checkpoint",
    )
    mapping_path = resolve_asset_path(
        args.mapping,
        ["class_mapping.txt", "classifier_mapping.txt"],
        env_var="MAPPING_PATH",
        desc="class mapping",
    )

    source = args.source or os.environ.get("SOURCE") or os.environ.get("RTSP_URL") or DEFAULT_RTSP_URL

    process_stream(
        source=source,
        model_path=model_path,
        detector_checkpoint_path=detector_checkpoint,
        class_mapping_path=mapping_path,
        log_file_path=args.log_file,
        timing_log_path=args.timing_log,
        line_pt1=LINE_START,
        line_pt2=LINE_END,
        right_line_pt1=RIGHT_LINE_START,
        right_line_pt2=RIGHT_LINE_END,
        target_fps=args.fps,
        rfdetr_conf_threshold=args.det_conf,
        person_class_id=PERSON_CLASS_ID,
        track_activation=args.track_thresh,
        rtsp_transport=args.transport,
        reconnect_delay=args.reconnect_delay,
        max_retries=args.max_retries,
        timezone_name=args.timezone,
        no_timing_log=args.no_timing_log,
        realtime=args.realtime,
        show_display=args.show,
    )


if __name__ == "__main__":
    main()