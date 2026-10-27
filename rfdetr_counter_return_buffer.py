#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuned RF-DETR Small + ByteTrack FORENSIC TIN/BOX COUNTER  -  v2
=======================================================
What changed vs. the original (see the long comment blocks marked  [FIX-n]):
  JITTER
  [FIX-1] RF-DETR was fed BGR frames; it expects RGB.
  [FIX-2] supervision's ByteTrack returns the RAW detector boxes (not the Kalman
          boxes), so every bit of detector noise reached the logic and the video.
          -> every track now goes through a per-track One-Euro smoother.
  [FIX-3] RF-DETR ran at conf 0.40, which starves ByteTrack's 2nd (low-score)
          association. -> detector runs low, ByteTrack decides (activation 0.40).
  [FIX-4] DETR has no NMS -> duplicate boxes on one person -> ID flicker.
          -> class-agnostic NMS before tracking.
  [FIX-5] ByteTrack was built with default frame_rate=30 although we feed \\\\~10 fps.
  [FIX-6] RTSP: the same frame could be processed twice / irregular dt.
          -> only fresh frames are processed, smoother uses real timestamps.
  COUNTING
  [FIX-7] Crossing direction was gated on whole-trajectory *horizontal* motion.
          Your lines are steeply slanted, so a person walking "up" the screen
          crosses a line with dx \\\\~ 0 (or the "wrong" sign) and was REJECTED,
          and the side memory was then overwritten so the crossing was lost
          forever.  -> direction now comes from the geometry of the crossing
          itself (inside -> outside of the work area).
  [FIX-8] Jitter around a line is handled with a hysteresis band instead of a
          bare sign flip.
  [FIX-9] If the ID changes while the person is at the line (fragmentation),
          the new ID inherits the old state (conservative stitching).
  [FIX-10] The "lock" counter was advanced on stale predictions (no new ResNet
          vote) -> "LOCKED full" without 5 real votes.  Fixed.
  [FIX-11] Duplicate-suppressed tracks were flagged counted=True. Fixed.
  [FIX-12] Classification cadence used the raw frame counter (frame_skip /
          interval aliasing, RTSP duplicates).  -> per-track age.
  [FIX-13] ResNet is now run once per frame on a batch of crops.
  [NEW]    \\\\<prefix>_count_events.csv : every crossing / count decision with the
           reason, and a Reason column in the summary CSV.
Everything else (ResNet152 classification, 0/2 values, deferred counting,
duplicate suppression, outputs, Drive sync, RTSP) is kept.
"""
# Install separately in Colab: pip install "rfdetr[plus]" "supervision>=0.25,<0.31" opencv-python
import os
import sys
import math
import csv
import json
import time
import shutil
import threading
import argparse
import subprocess
from collections import Counter, deque
# ================================================================
# 1. DEPENDENCIES
# ================================================================
def _ensure_deps():
    need = False
    try:
        import rfdetr  # noqa: F401
        import supervision as _sv
        if not hasattr(_sv, "ByteTrack"):
            need = True
    except ImportError:
        need = True
    if need:
        # sv.ByteTrack is deprecated and REMOVED in supervision 0.31 -> pin below it.
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-U",
             "rfdetr[plus]", "supervision>=0.25,<0.31", "opencv-python"],
            check=False,
        )
_ensure_deps()
import cv2
import numpy as np
import torch
import torch.nn as nn
import supervision as sv
from torchvision import transforms, models
from rfdetr import RFDETRSmall
if not hasattr(sv, "ByteTrack"):
    raise SystemExit(
        "This supervision version has no sv.ByteTrack (removed in 0.31).\n"
        "Run:  pip install 'supervision<0.31'   and restart the runtime."
    )
# ================================================================
# 2. GOOGLE DRIVE
# ================================================================
try:
    from google.colab import drive
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")
except ImportError:
    pass
# ================================================================
# 3. TUNABLES
# ================================================================
PERSON_CLASS_ID = 0          # Single-class person-only student usually uses class index 0; configurable via CLI
FULL_VALUE = 2               # a "full" person counts 2 tins/boxes
# ---- original parameters -----------------------------------------
LOCK_AFTER_FRAMES = 5        # consecutive "full" votes to lock
# ---- new / robustness parameters ---------------------------------
VOTE_WINDOW = 9              # majority is taken over the last N votes
LINE_HYSTERESIS_PX = 12      # dead-band either side of a line (replaces the unused BUFFER_ZONE_PX)
RETURN_RESET_DISTANCE_PX = 45   # preserve original-ID behavior
REASSIGNED_RETURN_RESET_DISTANCE_PX = 150  # reassign only after returning 150 px into work zone
RETURN_RESET_SECONDS = 0.8      # must remain deep inside this long to confirm a genuine return
DIR_WINDOW = 5               # smoothed points used for the movement direction
DIR_MIN_DISP = 8             # px over that window to call a direction
STITCH_GAP_SEC = 0.5         # only bridge very short tracker gaps
STITCH_MIN_RADIUS = 25       # px; conservative to reduce cross-person state transfer
STITCH_RADIUS_FACTOR = 0.6   # x box height
STITCH_BORDER = 30           # tracks that vanished near the frame edge "left the scene"
# ================================================================
# 4. ONE-EURO SMOOTHER  [FIX-2]
# ================================================================
class OneEuro:
    """
    Adaptive low-pass filter: heavy smoothing when the person is (almost) still
    (kills detector jitter), light smoothing when moving fast (no lag).
    Works on a numpy vector, uses real timestamps.
    """
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
# 5. DIRECTION / GEOMETRY
# ================================================================
def get_direction_label(dx, dy, threshold=5):
    """Original direction rule (unchanged)."""
    distance = math.hypot(dx, dy)
    if distance < threshold:
        return None
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360
    if 157.5 <= angle < 292.5:
        return "to_left"
    return "to_right"
def window_direction(hist, window=DIR_WINDOW, thr=DIR_MIN_DISP):
    """
    Direction from the net displacement over the last `window` SMOOTHED points
    instead of the last two raw centres (which flip randomly with jitter).
    """
    if len(hist) < 2:
        return None
    pts = list(hist)[-window:]
    return get_direction_label(
        pts[-1][0] - pts[0][0],
        pts[-1][1] - pts[0][1],
        threshold=thr,
    )
def line_side_sign(cx, cy, pt1, pt2):
    """Original cross-product sign helper (kept for reference/compat)."""
    x1, y1 = pt1
    x2, y2 = pt2
    val = (cx - x1) * (y2 - y1) - (cy - y1) * (x2 - x1)
    return 1 if val > 0 else (-1 if val < 0 else 0)
def signed_dist(px, py, p1, p2):
    """Signed perpendicular distance in PIXELS (same sign convention as line_side_sign)."""
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1) or 1.0
    return ((px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)) / length
class LineGate:
    """
    A counting line with an explicit notion of INSIDE (the work area between the
    two lines) and OUTSIDE.  A valid count is an INSIDE -> OUTSIDE crossing.
    Because "inside" is derived from where the other line is, the logic works for
    any slant / point order of the lines.  [FIX-7]
    """
    def __init__(self, name, p1, p2, inside_ref):
        self.name = name
        self.p1 = p1
        self.p2 = p2
        d = signed_dist(inside_ref[0], inside_ref[1], p1, p2)
        self.inside_sign = 1.0 if d >= 0 else -1.0
    def inside_dist(self, px, py):
        """>0 inside the work area, <0 outside (pixels from the line)."""
        return signed_dist(px, py, self.p1, self.p2) * self.inside_sign
def update_line_side(ls, gate, px, py, margin):
    """
    Hysteresis side tracker.  [FIX-8]
      returns "exit"   : committed side went inside -> outside   (countable)
              "return" : committed side went outside -> inside
              None     : nothing (including jitter inside the dead-band)
    The initial side is taken from the very first observation, so a track that
    appears right next to the line still produces a crossing when it moves on.
    """
    d = gate.inside_dist(px, py)
    if ls["side"] is None:
        ls["side"] = 1 if d >= 0 else -1
        return None
    if ls["side"] == 1 and d < -margin:
        ls["side"] = -1
        return "exit"
    if ls["side"] == -1 and d > margin:
        ls["side"] = 1
        return "return"
    return None
# ================================================================
# 6. CLASS MAPPING
# ================================================================
def load_class_mapping(mapping_path):
    mapping = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            for line in f:
                if ":" in line:
                    idx, cls_name = line.strip().split(":", 1)
                    mapping[int(idx)] = cls_name
    else:
        mapping = {0: "empty", 1: "full"}
    return mapping
# ================================================================
# 7. TRACK STATE + CLASS VOTING
# ================================================================
def make_fresh_state(tid, smooth_cutoff, smooth_beta, count_id):
    return {
        "ids": [tid],                       # id chain if stitched
        "smoother": OneEuro(smooth_cutoff, smooth_beta),
        "hist": deque(maxlen=12),           # smoothed anchor points
        "age": 0,
        "predictions": [],
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
        "reassigned_rearm_required": False,
        "rearm_buffer_side": None,
        "rearm_buffer_frames": 0,
        "cross_info": {},                   # side -> (x, y, processed_frame)
        "lines": {
            "left": {"side": None, "crossed": False, "pending": False},
            "right": {"side": None, "crossed": False, "pending": False},
        },
        "zone": None,
        "seen_work_zone": False,  # observed between/inside the counting lines before crossing
        "color": (0, 0, 255),
        "last_seen_frame": 0,
        "last_xy": None,
        "last_box": None,
        "last_size": None,
        "vel": (0.0, 0.0),
    }
def update_class(state, pred):
    """
    Add ONE new ResNet vote and update lock / majority.
    [FIX-10] The original re-ran the lock counter on the previous vote every time
    the interval fired, even with no new vote.
    """
    state["predictions"].append(pred)
    if not state["class_locked"]:
        if pred == state["last_consecutive_class"]:
            state["consecutive_same"] += 1
        else:
            state["consecutive_same"] = 1
            state["last_consecutive_class"] = pred
        if state["consecutive_same"] >= LOCK_AFTER_FRAMES and pred == "full":
            state["class_locked"] = True
            state["class_name"] = "full"
            state["class_value"] = FULL_VALUE
            return
        window = state["predictions"][-VOTE_WINDOW:]
        counts = Counter(window)
        majority = max(counts, key=lambda k: (counts[k], k == pred))
        state["class_name"] = majority
        if majority == "full":
            state["class_value"] = FULL_VALUE
        elif majority == "empty":
            state["class_value"] = 0
def not_counted_reason(s):
    if s["counted"]:
        return "COUNTED_FULL" if s["counted_value"] else "COUNTED_AS_EMPTY"
    if any(s["lines"][k]["pending"] for k in s["lines"]):
        return "CROSSED_BUT_CLASS_UNKNOWN"
    if not any(s["lines"][k]["crossed"] for k in s["lines"]):
        return "NEVER_CROSSED_EXIT_LINE"
    return "CROSSED_NOT_COUNTED"
# ================================================================
# 8. RTSP / FILE CAPTURE
# ================================================================
class RTSPCapture:
    """
    Threaded capture for RTSP (TCP, no buffering, auto-reconnect) or plain file.
    [FIX-6] read_latest() also returns a frame id so the main loop never
    processes the same frame twice.
    """
    def __init__(self, src, rtsp_transport="tcp", reconnect_delay=2.0, max_retries=10):
        self.src = src
        self.rtsp_transport = rtsp_transport
        self.reconnect_delay = reconnect_delay
        self.max_retries = max_retries
        self.is_stream = isinstance(src, int) or (
            isinstance(src, str) and src.startswith(("rtsp://", "rtsps://", "http://", "https://"))
        )
        self.cap = None
        self.frame = None
        self.ret = False
        self.stopped = False
        self.lock = threading.Lock()
        self.frame_id = 0
        if self.is_stream and isinstance(self.src, str) and self.src.startswith(("rtsp://", "rtsps://")):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self.rtsp_transport}|fflags;nobuffer|flags;low_delay|max_delay;500000"
            )
        self._connect()
        if self.is_stream and self.cap and self.cap.isOpened():
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
    def _connect(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        if self.is_stream and isinstance(self.src, str) and self.src.startswith(("rtsp://", "rtsps://")):
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(self.src)
        if self.cap.isOpened():
            if self.is_stream:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret, self.frame = ret, frame
                    self.frame_id += 1
    def _update(self):
        failures = 0
        while not self.stopped:
            if not self.cap or not self.cap.isOpened():
                if failures < self.max_retries:
                    failures += 1
                    print(f"⚠️ RTSP stream disconnected. Reconnecting ({failures}/{self.max_retries})...")
                    time.sleep(self.reconnect_delay)
                    self._connect()
                    continue
                print("❌ RTSP reconnect retries exhausted.")
                with self.lock:
                    self.ret = False
                break
            ret, frame = self.cap.read()
            if ret:
                failures = 0
                with self.lock:
                    self.ret, self.frame = True, frame
                    self.frame_id += 1
            else:
                failures += 1
                time.sleep(0.05)
                if failures >= 20:
                    print(f"⚠️ Failed to read RTSP frames ({failures}). Reconnecting...")
                    self._connect()
                    failures = 0
    def read(self):
        if self.is_stream:
            with self.lock:
                return self.ret, (self.frame.copy() if self.frame is not None else None)
        return self.cap.read()
    def read_latest(self):
        """(ret, frame, frame_id) - streams only."""
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None), self.frame_id
    def grab(self):
        return self.cap.grab() if self.cap is not None else False
    def isOpened(self):
        if self.is_stream:
            return not self.stopped and self.ret
        return self.cap is not None and self.cap.isOpened()
    def get(self, prop_id):
        if self.cap is not None:
            val = self.cap.get(prop_id)
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH and (val <= 0 or np.isnan(val)) and self.frame is not None:
                return float(self.frame.shape[1])
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT and (val <= 0 or np.isnan(val)) and self.frame is not None:
                return float(self.frame.shape[0])
            return val
        return 0.0
    def set(self, prop_id, value):
        return self.cap.set(prop_id, value) if self.cap else False
    def release(self):
        self.stopped = True
        if self.cap is not None:
            self.cap.release()
# ================================================================
# 9. RESNET BATCH CLASSIFIER  [FIX-13]
# ================================================================
def classify_crops(resnet, transform, crops_bgr, device, class_mapping):
    """One forward pass for all crops of the frame. -> [(class_name_lower, conf)]"""
    tensors = [transform(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(resnet(batch), dim=1)
        conf, pred = torch.max(probs, 1)
    return [
        (class_mapping.get(int(p), "Unknown").lower(), float(c))
        for p, c in zip(pred.cpu().tolist(), conf.cpu().tolist())
    ]
# ================================================================
# 10. MAIN PROCESSING
# ================================================================
def process_video(
    video_path,
    output_dir_path,
    output_prefix,
    model_path,
    detector_checkpoint_path,
    class_mapping_path,
    line_pt1,
    line_pt2,
    right_line_pt1,
    right_line_pt2,
    classify_interval=5,
    target_fps=10,
    rfdetr_conf_threshold=0.20,     # detector floor; ByteTrack decides below
    person_class_id=0,
    track_activation=0.40,          # == your old 0.4, but now used by the tracker
    lost_sec=3.0,
    min_consecutive=2,
    nms_iou=0.6,
    smooth_cutoff=0.6,
    smooth_beta=0.01,
    line_hysteresis=LINE_HYSTERESIS_PX,
    anchor="center",                # "center" (your original) or "bottom" (feet)
    enable_stitch=True,
    start_time_str=None,
    end_time_str=None,
    show_display=False,
    rtsp_transport="tcp",
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    is_stream = isinstance(video_path, int) or (
        isinstance(video_path, str) and video_path.startswith(("rtsp://", "rtsps://", "http://", "https://"))
    )
    if not is_stream and not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    if not os.path.exists(model_path):
        print(f"❌ ResNet model not found: {model_path}")
        return
    os.makedirs(output_dir_path, exist_ok=True)
    out_video = os.path.join(output_dir_path, f"{output_prefix}_video.mp4")
    out_summary_csv = os.path.join(output_dir_path, f"{output_prefix}_summary.csv")
    out_frame_csv = os.path.join(output_dir_path, f"{output_prefix}_per_frame_details.csv")
    out_events_csv = os.path.join(output_dir_path, f"{output_prefix}_count_events.csv")
    out_json = os.path.join(output_dir_path, f"{output_prefix}_telemetry_dump.json")
    out_img = os.path.join(output_dir_path, f"{output_prefix}_trajectory_map.jpg")
    is_drive_path = output_dir_path.startswith("/content/drive")
    def get_temp(path):
        return os.path.join("/content", "temp_" + os.path.basename(path)) if is_drive_path else path
    tmp_video, tmp_summary_csv, tmp_frame_csv = get_temp(out_video), get_temp(out_summary_csv), get_temp(out_frame_csv)
    tmp_events_csv, tmp_json, tmp_img = get_temp(out_events_csv), get_temp(out_json), get_temp(out_img)
    # ---------------- models ----------------
    class_mapping = load_class_mapping(class_mapping_path)
    print("🧠 Loading ResNet152 model...")
    resnet = models.resnet152(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, len(class_mapping))
    resnet.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    resnet = resnet.to(device)
    resnet.eval()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print(f"🎯 Loading fine-tuned RF-DETR Small checkpoint: {detector_checkpoint_path}")
    if not detector_checkpoint_path or not os.path.isfile(detector_checkpoint_path):
        raise FileNotFoundError(
            f"Fine-tuned RF-DETR Small checkpoint not found: {detector_checkpoint_path}\n"
            "Pass --detector-checkpoint with the path to your trained student checkpoint."
        )
    try:
        # Restore the fine-tuned RF-DETR Small architecture/config from the checkpoint.
        detector = RFDETRSmall.from_checkpoint(detector_checkpoint_path)
    except Exception as e:
        raise RuntimeError(
            "Could not load the fine-tuned RF-DETR Small checkpoint. "
            "Use a compatible RF-DETR Small checkpoint (e.g. checkpoint_best_ema.pth "
            "or the best_model.pth copied from that checkpoint) and a compatible rfdetr version. "
            f"Original error: {e}"
        ) from e
    print("✅ Fine-tuned RF-DETR Small loaded.")
    # ---------------- video ----------------
    if is_stream:
        print(f"📡 Connecting to RTSP stream: {video_path}")
    cap = RTSPCapture(video_path, rtsp_transport=rtsp_transport)
    if not cap.isOpened():
        print(f"❌ Could not open video or RTSP stream: {video_path}")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120 or np.isnan(fps):
        fps = float(target_fps) if target_fps > 0 else 25.0
    print(f"📹 Stream properties: {w}x{h} @ {fps:.1f} FPS")
    def parse_t(s):
        if ":" in s:
            m, sec = s.split(":")
            return int(m) * 60 + int(sec)
        return int(s)
    start_frame = 0
    end_frame = float("inf")
    max_duration_sec = None
    if is_stream:
        if start_time_str:
            print("ℹ️ Live RTSP stream: start time seeking skipped.")
        if end_time_str:
            max_duration_sec = parse_t(end_time_str)
            print(f"⏱️ Stream will process for {max_duration_sec}s")
    else:
        if start_time_str:
            start_frame = int(parse_t(start_time_str) * fps)
            print(f"⏱️ Starting from frame {start_frame} ({start_time_str})")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if end_time_str:
            end_frame = int(parse_t(end_time_str) * fps)
            print(f"⏱️ Ending at frame {end_frame} ({end_time_str})")
    if is_stream:
        frame_skip = 1
        effective_fps = float(target_fps) if target_fps > 0 else fps
    else:
        frame_skip = max(1, round(fps / target_fps))
        effective_fps = fps / frame_skip
    # ---------------- tracker  [FIX-3][FIX-5] ----------------
    trk_fps = max(1, int(round(effective_fps)))
    lost_buf = int(round(lost_sec * 30))          # supervision scales this by frame_rate/30
    max_lost_frames = int(trk_fps / 30.0 * lost_buf)
    EVICT_AFTER_FRAMES = max_lost_frames + 5      # always > ByteTrack's own memory
    stitch_gap = max(2, int(round(STITCH_GAP_SEC * effective_fps)))
    print("🎯 Initializing ByteTrack...")
    try:
        tracker = sv.ByteTrack(
            track_activation_threshold=track_activation,
            lost_track_buffer=lost_buf,
            minimum_matching_threshold=0.8,
            frame_rate=trk_fps,
            minimum_consecutive_frames=min_consecutive,
        )
    except TypeError:       # very old supervision
        tracker = sv.ByteTrack(
            track_thresh=track_activation,
            track_buffer=lost_buf,
            match_thresh=0.8,
            frame_rate=trk_fps,
        )
    print(f"✅ ByteTrack: frame_rate={trk_fps}, memory≈{max_lost_frames} frames "
          f"(\\\\~{max_lost_frames / effective_fps:.1f}s), activation={track_activation}")
    # ---------------- writers ----------------
    out = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), effective_fps, (w, h))
    if not out.isOpened():
        raise RuntimeError("❌ Could not initialize VideoWriter.")
    f_summary = open(tmp_summary_csv, "w", newline="", encoding="utf-8")
    w_summary = csv.writer(f_summary)
    w_summary.writerow([
        "Track_ID", "Start_Frame", "End_Frame", "Final_Class", "Counted",
        "Total_Predictions", "Full_Pct", "Empty_Pct",
        "Counted_Value", "Crossed_Left", "Crossed_Right", "Reason", "ID_Chain",
    ])
    f_frame = open(tmp_frame_csv, "w", newline="", encoding="utf-8")
    w_frame = csv.writer(f_frame)
    w_frame.writerow([
        "Frame", "Track_ID", "CX", "CY", "BBox", "RFDETR_Conf", "Direction",
        "ResNet_Raw", "Majority_Class", "Is_Locked", "Counted_Status", "Raw_CX", "Raw_CY",
    ])
    f_events = open(tmp_events_csv, "w", newline="", encoding="utf-8")
    w_events = csv.writer(f_events)
    w_events.writerow([
        "Frame", "Processed_Frame", "Track_ID", "Line", "Event",
        "Class_Value", "Added", "Total", "Left", "Right", "Note",
    ])
    # ---------------- geometry ----------------
    lmid = ((line_pt1[0] + line_pt2[0]) / 2.0, (line_pt1[1] + line_pt2[1]) / 2.0)
    rmid = ((right_line_pt1[0] + right_line_pt2[0]) / 2.0, (right_line_pt1[1] + right_line_pt2[1]) / 2.0)
    inside_ref = ((lmid[0] + rmid[0]) / 2.0, (lmid[1] + rmid[1]) / 2.0)
    gates = {
        "left": LineGate("left", line_pt1, line_pt2, inside_ref),
        "right": LineGate("right", right_line_pt1, right_line_pt2, inside_ref),
    }
    print(f"📐 Work area reference point: ({inside_ref[0]:.0f}, {inside_ref[1]:.0f}). "
          f"A count = person leaves the work area through the LEFT or RIGHT line.")
    # ---------------- state ----------------
    track_state = {}
    retired_ids = set()
    counted_ids = set()  # tracker IDs that have actually contributed to the count
    count_id_counter = {"next": 1}

    def new_count_id():
        """Return a globally unique counting-cycle ID, independent of the tracker ID."""
        value = f"C{count_id_counter['next']:07d}"
        count_id_counter["next"] += 1
        return value
    telemetry = {"tracks": {}}
    tally = {"total": 0, "left": 0, "right": 0}
    frame_count = start_frame
    processed_frame_count = 0
    bg_frame = None
    LEFT_LINE_COLOR = (255, 255, 0)
    RIGHT_LINE_COLOR = (255, 0, 255)
    GREEN, ORANGE = (0, 255, 0), (0, 165, 255)
    # ---------------- helpers ----------------
    def log_event(fc, pf, tid, line, event, val, added, note=""):
        w_events.writerow([fc, pf, tid, line, event, "" if val is None else val, added,
                           tally["total"], tally["left"], tally["right"], note])
    def resolve_crossing(state, tid, side, fc, pf, deferred):
        """Count only a confirmed Working -> destination-zone trip, once per cycle."""
        val = state["class_value"]
        # Defence in depth: never count a crossing unless this cycle was observed
        # in the shared working zone and its anchor is currently in the destination zone.
        d_left = gates["left"].inside_dist(state["last_xy"][0], state["last_xy"][1]) if state.get("last_xy") else 0.0
        d_right = gates["right"].inside_dist(state["last_xy"][0], state["last_xy"][1]) if state.get("last_xy") else 0.0
        destination_reached = (d_left < -line_hysteresis) if side == "left" else (d_right < -line_hysteresis)
        if not state.get("seen_work_zone", False) or not destination_reached:
            log_event(fc, pf, tid, side, "COUNT_DEFERRED_UNTIL_DESTINATION_ZONE", val, 0,
                      f"count_id={state.get('count_id', 'UNKNOWN')}; seen_work_zone={state.get('seen_work_zone', False)}; destination_reached={destination_reached}")
            state["lines"][side]["pending"] = True
            return False
        assigned = state.get("assigned_line")
        if assigned is not None and assigned != side:
            log_event(fc, pf, tid, side, "IGNORED_OTHER_LINE_ALREADY_ASSIGNED", val, 0,
                      f"count_id={state.get('count_id', 'UNKNOWN')} assigned_line={assigned}")
            return
        state["assigned_line"] = side
        state["counted"] = True
        counted_ids.add(int(tid))
        state["counted_value"] = val
        state["crossed_side"] = side
        state["color"] = GREEN
        tally["total"] += val
        tally[side] += val
        ev = "COUNTED_DEFERRED" if deferred else "COUNTED"
        log_event(fc, pf, tid, side, ev, val, val, f"count_id={state.get('count_id', 'UNKNOWN')}")
        print(f"📊 [Frame {fc}] CountID {state.get('count_id', 'UNKNOWN')} {ev} {side.upper()} LINE (+{val}) "
              f"| L:{tally['left']} R:{tally['right']} Total:{tally['total']}")
        return True
    def reset_for_next_counting_cycle(state, tid, fc, pf, side):
        """Start a fresh counting cycle after the counted person returns inside."""
        previous_count_id = state.get("count_id", "UNKNOWN")
        state["count_id"] = new_count_id()
        state["predictions"].clear()
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
        state["reassigned_rearm_required"] = False
        state["rearm_buffer_side"] = None
        state["rearm_buffer_frames"] = 0
        state["assigned_line"] = None
        state["cross_info"] = {}
        for line_name, line_state in state["lines"].items():
            line_state["crossed"] = False
            line_state["pending"] = False
        state["color"] = (0, 0, 255)
        log_event(fc, pf, tid, side, "NEW_COUNT_ID", None, 0,
                  f"previous_count_id={previous_count_id}; new_count_id={state['count_id']}")
        print(f"🔄 [Frame {fc}] Track {tid} returned inside: new counting ID {state['count_id']}; class reset to Unknown")
    def write_summary(tid, s):
        preds = s["predictions"]
        n = len(preds)
        c = Counter(preds)
        pf_ = c.get("full", 0) / n * 100 if n else 0
        pe_ = c.get("empty", 0) / n * 100 if n else 0
        td = telemetry["tracks"].get(tid, {})
        w_summary.writerow([
            tid, td.get("start_frame", ""), td.get("end_frame", ""), s["class_name"], s["counted"], n,
            f"{pf_:.2f}%", f"{pe_:.2f}%",
            "" if s["counted_value"] is None else s["counted_value"],
            s["lines"]["left"]["crossed"], s["lines"]["right"]["crossed"],
            not_counted_reason(s), ">".join(str(i) for i in s["ids"]),
        ])
    def try_stitch(new_tid, ax, ay, sw, sh, pf, fc, active_now):
        """[FIX-9] Re-attach a brand-new ID to a recently lost, nearby, same-size track."""
        best, best_d = None, 1e18
        for otid, ost in track_state.items():
            if otid in active_now or ost["last_xy"] is None:
                continue
            # Never transfer a counting/classification state from a track that has
            # already interacted with a counting line. In crowded footage this can
            # attach a different worker to a previously counted cycle (e.g. C0000028),
            # causing a false late-FULL upgrade and blocking the new worker at the
            # opposite line. A stitch is permitted only before any crossing/count.
            if (ost.get("counted", False)
                    or ost.get("assigned_line") is not None
                    or ost.get("crossed_side") in ("left", "right")
                    or any(ls.get("crossed", False) or ls.get("pending", False)
                           for ls in ost.get("lines", {}).values())
                    or ost.get("class_locked", False)):
                continue
            gap = pf - ost["last_seen_frame"]
            if gap < 1 or gap > stitch_gap:
                continue
            lb = ost["last_box"]
            if lb[0] < STITCH_BORDER or lb[1] < STITCH_BORDER or lb[2] > w - STITCH_BORDER or lb[3] > h - STITCH_BORDER:
                continue                               # it walked out of the scene
            pg = min(gap, 5)
            px = ost["last_xy"][0] + ost["vel"][0] * pg
            py = ost["last_xy"][1] + ost["vel"][1] * pg
            d = math.hypot(ax - px, ay - py)
            lw, lh = ost["last_size"]
            ratio = (sw * sh) / max(lw * lh, 1.0)
            # Tighten the spatial gate: the previous 0.6 x box-height radius
            # was too permissive for this crowded camera and allowed cross-person
            # state inheritance. Only stitch near-identical, short-gap detections.
            stitch_radius = max(STITCH_MIN_RADIUS, 0.25 * lh)
            if d <= stitch_radius and 0.65 <= ratio <= 1.6 and d < best_d:
                best, best_d = otid, d
        if best is None:
            return None
        st = track_state.pop(best)
        st["ids"].append(new_tid)
        retired_ids.add(best)
        old_tel = telemetry["tracks"].pop(best, None)
        if old_tel is not None:
            old_tel.setdefault("merged_ids", []).append(best)
            telemetry["tracks"][new_tid] = old_tel
        log_event(fc, pf, new_tid, "-", "STITCHED", st["class_value"], 0, f"inherited from ID {best} (d={best_d:.0f}px)")
        print(f"🔗 [Frame {fc}] ID {new_tid} stitched to lost ID {best} (d={best_d:.0f}px)")
        return st
    def detect_persons(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)          # [FIX-1] RF-DETR wants RGB
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
            pd = pd.with_nms(threshold=nms_iou, class_agnostic=True)   # [FIX-4]
        return pd
    # ============================================================
    # MAIN LOOP
    # ============================================================
    t_start = time.monotonic()
    last_fid = -1
    try:
        while cap.isOpened():
            if frame_count >= end_frame:
                print("🛑 Reached requested end time.")
                break
            if max_duration_sec is not None and time.monotonic() - t_start >= max_duration_sec:
                print("🛑 Reached requested duration.")
                break
            # ---------- read ----------
            if is_stream:
                ret, frame, fid = cap.read_latest()
                if not ret:
                    break
                if fid == last_fid:                      # [FIX-6] nothing new yet
                    time.sleep(0.005)
                    continue
                last_fid = fid
                frame_count += 1
                ts = time.monotonic()
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                ts = frame_count / fps
            if bg_frame is None:
                bg_frame = frame.copy()
            processed_frame_count += 1
            pf = processed_frame_count
            fc = frame_count
            # ---------- detect + track ----------
            person_dets = detect_persons(frame)
            tracked = tracker.update_with_detections(person_dets)
            items = []
            crops, crop_for = [], []
            # ======================================================
            # PHASE 1 : per-track update (smoothing, state, crop collection)
            # ======================================================
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
                    # ---- state (new / stitched) ----
                    state = track_state.get(tid)
                    if state is None:
                        if enable_stitch:
                            state = try_stitch(tid, rcx, rcy, rw, rh, pf, fc, active_now)
                        if state is None:
                            state = make_fresh_state(tid, smooth_cutoff, smooth_beta, new_count_id())
                        track_state[tid] = state
                    if tid not in telemetry["tracks"]:
                        telemetry["tracks"][tid] = {
                            "start_frame": pf, "end_frame": pf, "counted": False,
                            "final_class": "Unknown", "trajectory": [], "frame_logs": [],
                        }
                    t_data = telemetry["tracks"][tid]
                    # ---- [FIX-2] smooth centre + size ----
                    scx, scy, sw, sh = state["smoother"]([rcx, rcy, rw, rh], ts)
                    sw, sh = max(sw, 2.0), max(sh, 2.0)
                    sx1 = max(0.0, scx - sw / 2.0)
                    sy1 = max(0.0, scy - sh / 2.0)
                    sx2 = min(float(w), scx + sw / 2.0)
                    sy2 = min(float(h), scy + sh / 2.0)
                    ax = scx
                    ay = scy if anchor == "center" else sy2
                    # ---- velocity (for stitching) ----
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
                    t_data["end_frame"] = pf
                    t_data["trajectory"].append((round(ax, 1), round(ay, 1), pf))
                    # ---- direction / what to do this frame ----
                    direction = window_direction(state["hist"])
                    pending_any = state["lines"]["left"]["pending"] or state["lines"]["right"]["pending"]
                    due = (
                        state["age"] % classify_interval == 0
                        or state["class_name"] == "Unknown"
                        or pending_any
                    )
                    returning = (
                        (state["zone"] == "A" and direction == "to_right")
                        or (state["zone"] == "C" and direction == "to_left")
                    )
                    action, crop_idx = None, None
                    if due:
                        if returning and not pending_any:
                            action = "returning"
                        elif direction in ("to_left", "to_right") or pending_any:
                            pad = 5
                            cx1, cy1 = max(0, int(rx1) - pad), max(0, int(ry1) - pad)
                            cx2, cy2 = min(w, int(rx2) + pad), min(h, int(ry2) + pad)
                            crop = frame[cy1:cy2, cx1:cx2]        # raw box = same as training crops
                            if crop.size > 0:
                                action = "classify"
                                crop_idx = len(crops)
                                crops.append(crop)
                    items.append({
                        "tid": tid, "state": state, "t_data": t_data, "conf": float(conf),
                        "sbox": (int(sx1), int(sy1), int(sx2), int(sy2)),
                        "ax": ax, "ay": ay, "raw": (rcx, rcy),
                        "direction": direction, "action": action, "crop_idx": crop_idx,
                    })
            # ======================================================
            # PHASE 2 : one batched ResNet pass
            # ======================================================
            results = classify_crops(resnet, transform, crops, device, class_mapping) if crops else []
            # ======================================================
            # PHASE 3 : classification result, counting, logging, drawing
            # ======================================================
            for it in items:
                tid, state, t_data = it["tid"], it["state"], it["t_data"]
                ax, ay = it["ax"], it["ay"]
                direction = it["direction"]
                # ---- classification vote ----
                resnet_raw = "N/A"
                if it["action"] == "returning":
                    # Returning to the work zone starts a new counting cycle; never force class to empty.
                    resnet_raw = "new counting cycle reset (no empty vote)"
                elif it["crop_idx"] is not None:
                    name, c = results[it["crop_idx"]]
                    if name != "unknown":
                        update_class(state, name)
                        resnet_raw = f"{name} ({c * 100:.1f}%)"
                # ---- zone first: all count decisions use the CURRENT position ----
                dL = gates["left"].inside_dist(ax, ay)
                dR = gates["right"].inside_dist(ax, ay)
                state["zone"] = "A" if dL < 0 else ("C" if dR < 0 else "B")
                # ---- deferred resolution ----
                # Classification may become available while the person is still in the
                # working zone. Keep the event pending until that same track actually
                # reaches its assigned destination zone. If they return instead, the
                # normal return handler cancels the pending crossing without counting.
                for side in ("left", "right"):
                    ls = state["lines"][side]
                    destination_reached = (dL < -line_hysteresis) if side == "left" else (dR < -line_hysteresis)
                    if ls["pending"] and state["class_value"] is not None and destination_reached:
                        if state["counted"]:
                            ls["pending"] = False
                            log_event(fc, pf, tid, side, "IGNORED_ALREADY_COUNTED", state["class_value"], 0)
                        else:
                            # resolve_crossing validates working-zone history and destination again.
                            resolve_crossing(state, tid, side, fc, pf, deferred=True)
                            if state.get("counted", False):
                                ls["pending"] = False
                # Remember that this counting cycle was observed in the working zone.
                # This is needed for the late-classification edge case: it may cross
                # while temporarily classified empty, then later lock as FULL.
                if dL >= 0 and dR >= 0:
                    state["seen_work_zone"] = True

                # A late FULL lock may upgrade an EMPTY count only while this SAME
                # track is physically in the corresponding destination zone. A FULL
                # prediction in the shared working zone must never increment either side.
                late_side_candidate = state.get("crossed_side")
                late_destination_reached = (
                    (late_side_candidate == "left" and dL < -line_hysteresis)
                    or (late_side_candidate == "right" and dR < -line_hysteresis)
                )
                if (state.get("seen_work_zone", False)
                        and late_destination_reached
                        and late_side_candidate in ("left", "right")
                        and state.get("counted", False)
                        and state.get("counted_value") == 0
                        and state.get("class_locked", False)
                        and state.get("class_value") == FULL_VALUE):
                    late_side = late_side_candidate
                    # Late FULL classification updates only the line that this
                    # counting cycle crossed; do not increment the global total here.
                    tally[late_side] += FULL_VALUE
                    state["counted_value"] = FULL_VALUE
                    state["color"] = GREEN
                    log_event(fc, pf, tid, late_side, "LATE_FULL_LOCK_UPGRADE", FULL_VALUE, FULL_VALUE,
                              f"count_id={state.get('count_id', 'UNKNOWN')}; crossed_from_work_zone=True; full_locked=True")
                    print(f"⬆️ [Frame {fc}] CountID {state.get('count_id', 'UNKNOWN')} FULL LOCKED AFTER CROSSING "
                          f"| {late_side.upper()} line +{FULL_VALUE} | Total unchanged: {tally['total']}")

                # ---- confirmed-return buffer ----
                # Do NOT create a new counting ID merely because the centre jitters
                # a few pixels back across the line. A counted worker must move well
                # inside the work area and remain there for a short continuous period.
                # This prevents repeated +2 counts when the same worker throws multiple
                # tins while standing close to the line.
                # Re-arm only a tracker whose CURRENT cycle was actually counted.
                # The historical counted-ID set helps identify a tracker that has
                # counted before, but is not sufficient on its own: an uncounted cycle
                # that crosses/returns must keep its existing Count ID. After a counted
                # worker returns at least 150 px into the work zone continuously for
                # 0.8 s, assign a fresh Count ID so a newly carried box can be counted.
                previously_counted_track = int(tid) in counted_ids
                # Reassign a fresh Count ID only for a cycle that actually counted.
                # Merely crossing/returning with an unresolved or otherwise uncounted
                # cycle must never change its Count ID.
                current_cycle_was_counted = bool(state.get("counted", False))
                has_counted_crossing = state.get("crossed_side") in ("left", "right")
                if previously_counted_track and current_cycle_was_counted and has_counted_crossing:
                    # Symmetric for both directions: a LEFT-counted cycle must return
                    # deep inside from the left gate; a RIGHT-counted cycle must return
                    # deep inside from the right gate. Each track owns its own buffer,
                    # so simultaneous workers on opposite sides are handled independently.
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
                            log_event(fc, pf, tid, return_side, "RETURN_BUFFER_CONFIRMED", state.get("counted_value"), 0,
                                      f"inside_distance={return_dist:.1f}px; frames={state['return_buffer_frames']}; previously_counted_track=True; current_cycle_was_counted=True; fresh_counting_cycle_rearmed=True")
                            reset_for_next_counting_cycle(state, tid, fc, pf, return_side)
                    else:
                        state["return_buffer_side"] = None
                        state["return_buffer_frames"] = 0
                else:
                    state["return_buffer_side"] = None
                    state["return_buffer_frames"] = 0

                # ---- crossings [FIX-7][FIX-8] ----
                for side in ("left", "right"):
                    ls = state["lines"][side]
                    ev = update_line_side(ls, gates[side], ax, ay, line_hysteresis)
                    if ev == "exit":
                        if ls["crossed"]:
                            continue
                        ls["crossed"] = True
                        state["cross_info"][side] = (ax, ay, pf)
                        assigned = state.get("assigned_line")
                        destination_reached = (dL < -line_hysteresis) if side == "left" else (dR < -line_hysteresis)
                        if not state.get("seen_work_zone", False) or not destination_reached:
                            log_event(fc, pf, tid, side, "IGNORED_NOT_WORKING_TO_DESTINATION", state["class_value"], 0,
                                      f"count_id={state.get('count_id', 'UNKNOWN')}; seen_work_zone={state.get('seen_work_zone', False)}; destination_reached={destination_reached}")
                        elif assigned is not None and assigned != side:
                            log_event(fc, pf, tid, side, "IGNORED_OTHER_LINE_ALREADY_ASSIGNED", state["class_value"], 0,
                                      f"count_id={state.get('count_id', 'UNKNOWN')} assigned_line={assigned}")
                        elif state["counted"]:
                            log_event(fc, pf, tid, side, "IGNORED_ALREADY_COUNTED", state["class_value"], 0,
                                      f"count_id={state.get('count_id', 'UNKNOWN')}")
                        else:
                            # All counting cycles (including R1, R2, ...) use the same
                            # normal counting rules. If the class is known, resolve now;
                            # otherwise defer until classification becomes available.
                            if state["class_value"] is not None:
                                state["assigned_line"] = side
                                resolve_crossing(state, tid, side, fc, pf, deferred=False)
                            else:
                                if assigned is None:
                                    state["assigned_line"] = side
                                ls["pending"] = True
                                log_event(fc, pf, tid, side, "PENDING_CLASS_UNKNOWN", None, 0,
                                          f"count_id={state.get('count_id', 'UNKNOWN')}; class unknown")
                                print(f"⏳ [Frame {fc}] CountID {state.get('count_id', 'UNKNOWN')} crossed {side.upper()} LINE, class unknown - deferring")
                    elif ev == "return":
                        if ls["pending"] and not state["counted"]:
                            ls["pending"] = False
                            ls["crossed"] = False
                            if state.get("assigned_line") == side:
                                state["assigned_line"] = None
                            log_event(fc, pf, tid, side, "PENDING_CANCELLED_RETURNED", state["class_value"], 0)
                        elif state["counted"] and state.get("crossed_side") == side:
                            # A shallow return across the line is not enough to start a
                            # new counting cycle. The confirmed-return buffer above handles reset.
                            log_event(fc, pf, tid, side, "RETURN_SEEN_WAITING_FOR_BUFFER", state["counted_value"], 0,
                                      f"need >= {REASSIGNED_RETURN_RESET_DISTANCE_PX}px inside for {RETURN_RESET_SECONDS:.1f}s before assigning next counting cycle")
                        else:
                            log_event(fc, pf, tid, side, "RETURN_CROSSING_IGNORED", state["class_value"], 0,
                                      f"count_id={state.get('count_id', 'UNKNOWN')}")
                # ---- telemetry ----
                t_data["final_class"] = state["class_name"]
                t_data["counted"] = state["counted"]
                x1, y1, x2, y2 = it["sbox"]
                w_frame.writerow([
                    pf, tid, int(ax), int(ay), f"[{x1},{y1},{x2},{y2}]", f"{it['conf']:.2f}",
                    direction if direction else "N/A", resnet_raw,
                    state["class_name"], state["class_locked"], state["counted"],
                    int(it["raw"][0]), int(it["raw"][1]),
                ])
                t_data["frame_logs"].append({
                    "frame": pf, "cx": round(ax, 1), "cy": round(ay, 1), "bbox": [x1, y1, x2, y2],
                    "raw_cx": round(it["raw"][0], 1), "raw_cy": round(it["raw"][1], 1),
                    "rfdetr_conf": it["conf"], "direction": direction, "resnet_raw": resnet_raw,
                    "majority_class": state["class_name"], "is_locked": state["class_locked"],
                    "count_id": state.get("count_id", "UNKNOWN"),
                    "counted": state["counted"],
                })
                # ---- draw ----
                color = state["color"]
                pts = t_data["trajectory"]
                for i in range(1, len(pts)):
                    cv2.line(frame, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                             (int(pts[i][0]), int(pts[i][1])), color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (int(ax), int(ay)), 5, color, -1)
                dv = state["class_value"] if state["class_value"] is not None else "?"
                label = (f"CountID:{state.get('count_id', 'UNKNOWN')} (Track:{tid}) c:{it['conf']:.2f} {state['class_name']} (+{dv}) "
                         f"{'LOCKED' if state['class_locked'] else ''}")
                cv2.putText(frame, label, (x1, max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # ======================================================
            # STALE TRACKS
            # ======================================================
            stale = [tid for tid, s in track_state.items()
                     if pf - s["last_seen_frame"] > EVICT_AFTER_FRAMES]
            for tid in stale:
                s = track_state[tid]
                for side in ("left", "right"):
                    if s["lines"][side]["pending"]:
                        log_event(fc, pf, tid, side, "UNRESOLVED_AT_RETIRE", None, 0, "never got a class")
                write_summary(tid, s)
                retired_ids.add(tid)
                del track_state[tid]
            # ======================================================
            # OVERLAYS
            # ======================================================
            cv2.line(frame, line_pt1, line_pt2, LEFT_LINE_COLOR, 3)
            lmx, lmy = (line_pt1[0] + line_pt2[0]) // 2, (line_pt1[1] + line_pt2[1]) // 2
            cv2.putText(frame, f"L: {tally['left']}", (max(10, lmx - 80), lmy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, LEFT_LINE_COLOR, 3)
            cv2.line(frame, right_line_pt1, right_line_pt2, RIGHT_LINE_COLOR, 3)
            rmx, rmy = (right_line_pt1[0] + right_line_pt2[0]) // 2, (right_line_pt1[1] + right_line_pt2[1]) // 2
            cv2.putText(frame, f"R: {tally['right']}", (min(w - 10, rmx + 10), rmy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, RIGHT_LINE_COLOR, 3)
            cv2.putText(frame, f"Total: {tally['total']}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
            cv2.putText(frame, f"Left Line: {tally['left']}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, LEFT_LINE_COLOR, 3)
            cv2.putText(frame, f"Right Line: {tally['right']}", (50, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.2, RIGHT_LINE_COLOR, 3)
            out.write(frame)
            if show_display:
                try:
                    cv2.imshow("RF-DETR + ByteTrack Forensic Tin Counter", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\n⏹️ 'q' pressed. Stopping...")
                        break
                except Exception:
                    show_display = False
            if pf % 25 == 0:
                print(f"🎥 Frame: {fc} | Processed: {pf} | Active IDs: {len(track_state)} | "
                      f"Total Count: {tally['total']} (L: {tally['left']}, R: {tally['right']})")
            # skip frames cheaply for files (grab() does not decode)
            if not is_stream and frame_skip > 1:
                for _ in range(frame_skip - 1):
                    if not cap.grab():
                        break
                    frame_count += 1
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user (Ctrl+C). Finalizing forensic reports...")
    finally:
        cap.release()
        out.release()
        if show_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
    # ---------------- flush remaining ----------------
    for tid, s in track_state.items():
        for side in ("left", "right"):
            if s["lines"][side]["pending"]:
                log_event(frame_count, processed_frame_count, tid, side, "UNRESOLVED_AT_END", None, 0, "never got a class")
        write_summary(tid, s)
    f_summary.close()
    f_frame.close()
    f_events.close()
    # ---------------- trajectory map ----------------
    if bg_frame is not None:
        print("🗺️ Generating Global Trajectory Map Image...")
        img = bg_frame.copy()
        cv2.line(img, line_pt1, line_pt2, LEFT_LINE_COLOR, 3)
        cv2.line(img, right_line_pt1, right_line_pt2, RIGHT_LINE_COLOR, 3)
        for tid, tdata in telemetry["tracks"].items():
            color = GREEN if tdata["counted"] else (0, 0, 255)
            pts = tdata["trajectory"]
            for i in range(1, len(pts)):
                cv2.line(img, (int(pts[i - 1][0]), int(pts[i - 1][1])), (int(pts[i][0]), int(pts[i][1])), color, 2)
            if pts:
                cv2.putText(img, f"ID:{tid}", (int(pts[0][0]), int(pts[0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imwrite(tmp_img, img)
    print("📝 Saving Full JSON Telemetry Dump...")
    with open(tmp_json, "w") as f:
        json.dump(telemetry, f, indent=4)
    # ---------------- drive sync ----------------
    if is_drive_path:
        print("🚚 Moving all forensic files to Google Drive...")
        for tmp_path, real_path in [
            (tmp_video, out_video), (tmp_summary_csv, out_summary_csv), (tmp_frame_csv, out_frame_csv),
            (tmp_events_csv, out_events_csv), (tmp_json, out_json), (tmp_img, out_img),
        ]:
            if os.path.exists(tmp_path):
                try:
                    shutil.copy(tmp_path, real_path)
                    os.remove(tmp_path)
                except Exception as e:
                    print(f"❌ Failed to move {os.path.basename(real_path)}: {e}")
    print()
    print("=" * 70)
    print("✅ ALL FORENSIC FILES SAVED AND SYNCED!")
    print("=" * 70)
    print()
    print(f"🎥 Video: {out_video}")
    print(f"📊 Summary CSV: {out_summary_csv}")
    print(f"📄 Frame CSV: {out_frame_csv}")
    print(f"🧾 Count events CSV: {out_events_csv}")
    print(f"📝 JSON: {out_json}")
    print(f"🗺️ Trajectory Map: {out_img}")
    print()
    print(f"🔢 FINAL TOTAL COUNT: {tally['total']}")
    print(f"⬅️ LEFT LINE COUNT: {tally['left']}")
    print(f"➡️ RIGHT LINE COUNT: {tally['right']}")
    print(f"👤 TOTAL TRACKS: {len(telemetry['tracks'])}")
# ================================================================
# 11. MAIN
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuned RF-DETR Small + ByteTrack Tin Counter v2 (Video / RTSP)")
    parser.add_argument("--source", "-s", type=str, default=None, help="Video path or RTSP URL")
    parser.add_argument("--output-dir", "-o", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="forensic_run_cam3")
    parser.add_argument("--model", type=str, default=None, help="ResNet152 classifier weights")
    parser.add_argument("--detector-checkpoint", type=str, default=None, help="Path to fine-tuned RF-DETR Small checkpoint")
    parser.add_argument("--person-class-id", type=int, default=0, help="Person class index in the fine-tuned detector; usually 0 for a single-class model")
    parser.add_argument("--mapping", type=str, default=None, help="class mapping file")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--transport", type=str, default="tcp", choices=["tcp", "udp"])
    parser.add_argument("--duration", type=str, default=None, help="RTSP max duration (MM:SS or seconds)")
    parser.add_argument("--start", type=str, default=None, help="file start time (MM:SS or seconds)")
    parser.add_argument("--end", type=str, default=None, help="file end time (MM:SS or seconds)")
    # tuning
    parser.add_argument("--det-conf", type=float, default=0.20, help="RF-DETR floor (ByteTrack filters further)")
    parser.add_argument("--track-thresh", type=float, default=0.40, help="ByteTrack activation threshold")
    parser.add_argument("--smooth-cutoff", type=float, default=0.6, help="lower = smoother when still")
    parser.add_argument("--smooth-beta", type=float, default=0.01, help="higher = less lag when moving")
    parser.add_argument("--hyst", type=float, default=LINE_HYSTERESIS_PX, help="line dead-band in px")
    parser.add_argument("--anchor", choices=["center", "bottom"], default="center")
    parser.add_argument("--no-stitch", action="store_true", help="disable ID stitching")
    args, unknown = parser.parse_known_args()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOCAL_MODELS_DIR = os.path.join(BASE_DIR, "models")

    def resolve_path(cli_val, local_name, fallback_colab):
        if cli_val:
            return cli_val
        local_file = os.path.join(LOCAL_MODELS_DIR, local_name)
        if os.path.exists(local_file):
            return local_file
        return fallback_colab

    VIDEO_IN = (
        args.source
        or os.environ.get("RTSP_URL")
        or ("/content/drive/MyDrive/Amrendra_Guardex/tin_counter_new_angle/testing_video_folder/Saturday_10-Oct-2026_03-14-21_PM.mp4")
    )
    OUT_DIR = (
        args.output_dir
        or os.path.join(BASE_DIR, "output")
    )
    MODEL_WEIGHTS = resolve_path(
        args.model,
        "ResNet152_direction_best_4.pth",
        "/content/drive/MyDrive/Amrendra_Guardex/tin_counter_new_angle/model/ResNet152_direction_best_4.pth"
    )
    DETECTOR_CHECKPOINT = resolve_path(
        args.detector_checkpoint,
        "best_model.pth",
        "/content/drive/MyDrive/Amrendra_Guardex/tin_counter_new_angle/KD/runs (1)/student_train/best_model.pth"
    )
    MAPPING_FILE = (
        args.mapping
        or (os.path.join(LOCAL_MODELS_DIR, "classifier_mapping.txt") if os.path.exists(os.path.join(LOCAL_MODELS_DIR, "classifier_mapping.txt")) else None)
        or (os.path.join(LOCAL_MODELS_DIR, "class_mapping.txt") if os.path.exists(os.path.join(LOCAL_MODELS_DIR, "class_mapping.txt")) else None)
        or "/content/drive/MyDrive/Amrendra_Guardex/tin_counter_new_angle/dataset/class_mapping.txt"
    )
    LINE_START, LINE_END = (880, 1047), (1008, 4)
    RIGHT_LINE_START, RIGHT_LINE_END = (2069, 1126), (1482, 3)
    is_stream = isinstance(VIDEO_IN, str) and VIDEO_IN.startswith(("rtsp://", "rtsps://", "http://", "https://"))
    if is_stream:
        print(f"📡 RTSP stream configured: {VIDEO_IN}")
        start_time = None
        if args.duration:
            end_time = args.duration
        else:
            try:
                end_time = input("Enter duration to run (MM:SS or seconds), or Enter to run continuously: ").strip() or None
            except (EOFError, KeyboardInterrupt):
                end_time = None
    else:
        try:
            start_time = args.start if args.start is not None else (
                input("Enter start time (MM:SS or SS) or press Enter to start from beginning: ").strip() or None)
        except (EOFError, KeyboardInterrupt):
            start_time = None
        try:
            end_time = args.end if args.end is not None else (
                input("Enter end time (MM:SS or SS) or press Enter to process to the end: ").strip() or None)
        except (EOFError, KeyboardInterrupt):
            end_time = None
    process_video(
        video_path=VIDEO_IN,
        output_dir_path=OUT_DIR,
        output_prefix=args.prefix,
        model_path=MODEL_WEIGHTS,
        detector_checkpoint_path=DETECTOR_CHECKPOINT,
        class_mapping_path=MAPPING_FILE,
        line_pt1=LINE_START,
        line_pt2=LINE_END,
        right_line_pt1=RIGHT_LINE_START,
        right_line_pt2=RIGHT_LINE_END,
        classify_interval=5,
        target_fps=10,
        rfdetr_conf_threshold=args.det_conf,
        person_class_id=args.person_class_id,
        track_activation=args.track_thresh,
        smooth_cutoff=args.smooth_cutoff,
        smooth_beta=args.smooth_beta,
        line_hysteresis=args.hyst,
        anchor=args.anchor,
        enable_stitch=not args.no_stitch,
        start_time_str=start_time,
        end_time_str=end_time,
        show_display=args.show,
        rtsp_transport=args.transport,
    )
