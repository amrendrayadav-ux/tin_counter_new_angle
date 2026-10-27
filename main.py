# Standard-library imports are enough for count-log query mode.
# CV/model dependencies are imported lazily only when running inference.
import argparse
from collections import Counter, deque
import csv
import datetime
import json
import math
import os
import shutil
import time
import threading
import queue
import uuid
from zoneinfo import ZoneInfo

# =====================================================================
# 1. FIXED CAMERA GEOMETRY & VALIDATED HYPERPARAMETERS
# =====================================================================

# Target processing rate: 10 FPS. Frame-drop / skip logic maintains this
# cadence for both video files and live RTSP/webcam streams with zero FIFO backlog.
TARGET_PROCESS_FPS = 10.0
SAMPLE_PERIOD_SEC = 1.0 / TARGET_PROCESS_FPS
line_pt1 = (851, 1280)
line_pt2 = (1008, 4)

right_line_pt1 = (2020, 1280)
right_line_pt2 = (1482, 3)

# Shifted physical right exit line for display (the crossing algorithm still measures
# distance from the inner reference line above and commits at RIGHT_CROSSING_BUFFER_PX).
RIGHT_DRAW_LINE_START = (2099, 1280)
RIGHT_DRAW_LINE_END = (1561, 3)

# Asymmetric Entry-Trigger & Return-Buffer Thresholds (Fixed Camera Angle)
LEFT_CROSSING_BUFFER_PX = 12.0
RIGHT_CROSSING_BUFFER_PX = 85.0
LEFT_RETURN_DEADBAND_PX = 12.0
RIGHT_RETURN_DEADBAND_PX = 15.0
RETURN_BUFFER_PX = 150.0
MIN_WORK_ZONE_DEPTH_PX = 40.0

# Classification & Lock Validation Thresholds
LOCK_STREAK = 4
LOCK_MIN_CONF = 0.50
LOCK_MIN_MEAN_CONF = 0.68
LOCK_MIN_DISP_PX = 18.0

PRE_EMPTY_SKEPTICISM_VOTES = 4
PRE_EMPTY_MIN_DISP_PX = 50.0
PRE_EMPTY_MIN_CONF = 0.88

LOITER_EMPTY_VOTES = 10
LOITER_LOCK_MIN_CONF = 0.78
LOITER_LOCK_MEAN_CONF = 0.85
LOITER_LOCK_MIN_DISP_PX = 35.0

STALE_LOCK_FRAMES = 35
STALE_LOCK_EMPTY_VOTES = 5
STALE_LOCK_TRANSIT_PX = 250.0
TURNBACK_NEAR_LINE_PX = 80.0
TURNBACK_RETREAT_PX = 200.0


def _signed_line_distance(px, py, pt1, pt2):
  """Signed perpendicular distance (in pixels) from point (px, py) to line pt1->pt2."""
  x1, y1 = pt1
  x2, y2 = pt2
  dx = x2 - x1
  dy = y2 - y1
  length = math.hypot(dx, dy) or 1.0
  return ((px - x1) * dy - (py - y1) * dx) / length


def _compute_line_inside_sign(pt1, pt2, ref_point):
  """Returns +1.0 or -1.0 so that multiplying _signed_line_distance by this sign

  makes the 'inside' (work zone between the lines) strictly POSITIVE (+),
  and the 'outside' (counted exit zone) strictly NEGATIVE (-).
  """
  d = _signed_line_distance(ref_point[0], ref_point[1], pt1, pt2)
  return 1.0 if d >= 0 else -1.0


_left_mid = (
    (line_pt1[0] + line_pt2[0]) / 2.0,
    (line_pt1[1] + line_pt2[1]) / 2.0,
)
_right_mid = (
    (right_line_pt1[0] + right_line_pt2[0]) / 2.0,
    (right_line_pt1[1] + right_line_pt2[1]) / 2.0,
)
WORK_ZONE_REF = (
    (_left_mid[0] + _right_mid[0]) / 2.0,
    (_left_mid[1] + _right_mid[1]) / 2.0,
)

LEFT_INSIDE_SIGN = _compute_line_inside_sign(line_pt1, line_pt2, WORK_ZONE_REF)
RIGHT_INSIDE_SIGN = _compute_line_inside_sign(
    right_line_pt1, right_line_pt2, WORK_ZONE_REF
)


def get_inside_distance(point, pt1, pt2, inside_sign):
  """Returns signed distance in pixels:

  > 0  : Inside the central work zone
  == 0 : On the counting line
  < 0  : Outside the line (in the counted exit zone)
  """
  return _signed_line_distance(point[0], point[1], pt1, pt2) * inside_sign


def evaluate_buffer_crossing(
    prev_side,
    curr_dist,
    counted,
    return_armed,
    crossing_buffer_px,
    return_deadband_px,
    return_buffer_px=RETURN_BUFFER_PX,
):
  """Unified Entry-Trigger + Return-Buffer State Machine."""
  events = []

  if prev_side is None:
    if curr_dist >= return_deadband_px:
      init_side = 1
    elif curr_dist <= -crossing_buffer_px:
      init_side = -1
    else:
      init_side = None
    return init_side, return_armed, events

  curr_side = prev_side

  if prev_side == 1:
    if curr_dist <= -crossing_buffer_px:
      curr_side = -1
      events.append("crossed")
  elif prev_side == -1:
    if not counted:
      if curr_dist >= return_deadband_px:
        curr_side = 1
        events.append("return")
    else:
      if not return_armed and curr_dist >= return_deadband_px:
        return_armed = True
        events.append("return_seen")
      if return_armed and curr_dist >= return_buffer_px:
        curr_side = 1
        return_armed = False
        events.append("return_confirmed")

  return curr_side, return_armed, events


# =====================================================================
# 2. UNIFIED VOTE LOCKING & STALE/TURN-BACK EXPIRATION LOGIC
# =====================================================================
def evaluate_vote_locking(state, cls_name, conf_val, cx, cy, d_left, d_right):
  """Appends a classification vote and evaluates whether the track should lock

  into 'full' using:
    1) Work-Zone Presence Gate (must have entered work zone: d_left >= 0 and
    d_right >= 0)
    2) Base Kinematic & Confidence Gate (4 consecutive 'full', min_c >= 0.50,
    mean_c >= 0.68, disp >= 18px)
    3) Cycle-Wide Loitering Gate (if cycle_empty_votes >= 10, requires min_c >=
    0.78, mean_c >= 0.85, disp >= 35px)
    4) Pre-Lock Empty Skepticism Gate (if cycle_empty_votes >= 4, requires disp
    >= 50px or min_c >= 0.88)
  """
  state["predictions"].append(cls_name)
  state["vote_details"].append((cls_name, conf_val, cx, cy, d_left, d_right))

  in_wz = (d_left >= 0.0) and (d_right >= 0.0)
  if cls_name == "empty":
    state["cycle_empty_votes"] = state.get("cycle_empty_votes", 0) + 1
    if in_wz:
      state["wz_empty_votes"] = state.get("wz_empty_votes", 0) + 1

  # Gate 1: Never lock as 'full' while outside the work zone (prevents V3 Track 9 pre-entry lock)
  if not in_wz or not state.get("seen_work_zone", False):
    return

  preds = state["predictions"]
  details = state["vote_details"]

  if len(preds) >= LOCK_STREAK and all(
      p == "full" for p in preds[-LOCK_STREAK:]
  ):
    last4 = details[-LOCK_STREAK:]
    confs = [c for _, c, _, _, _, _ in last4]
    min_c = min(confs)
    mean_c = sum(confs) / float(LOCK_STREAK)
    disp_px = math.hypot(last4[-1][2] - last4[0][2], last4[-1][3] - last4[0][3])

    if (
        min_c >= LOCK_MIN_CONF
        and mean_c >= LOCK_MIN_MEAN_CONF
        and disp_px >= LOCK_MIN_DISP_PX
    ):
      cycle_empty = state.get("cycle_empty_votes", 0)

      if cycle_empty >= LOITER_EMPTY_VOTES:
        if (
            min_c >= LOITER_LOCK_MIN_CONF
            and mean_c >= LOITER_LOCK_MEAN_CONF
            and disp_px >= LOITER_LOCK_MIN_DISP_PX
        ):
          state["is_locked"] = True
          state["locked_class"] = "full"
      elif cycle_empty >= PRE_EMPTY_SKEPTICISM_VOTES:
        if disp_px >= PRE_EMPTY_MIN_DISP_PX or min_c >= PRE_EMPTY_MIN_CONF:
          state["is_locked"] = True
          state["locked_class"] = "full"
      else:
        state["is_locked"] = True
        state["locked_class"] = "full"

  if state.get("is_locked") and state.get("locked_pos") is None:
    state["locked_pos"] = (cx, cy, d_left, d_right)
    state["min_dL_since_lock"] = d_left
    state["min_dR_since_lock"] = d_right


def check_stale_lock_expiration(state, current_pf, cx, cy, d_left, d_right):
  """Expires an uncounted 'full' lock when:

  1) Transit-Aware Stale Lock: age > 35 frames AND wz_empty_votes >= 5 AND
     the worker has NOT advanced >= 250 px toward either exit line since locking
     (expires V2 Tracks 210 & 907 while preserving V3 Track 1958).
  2) Turn-Back / Aborted Delivery Expiration: the worker approached within 80 px
     of an exit line after locking, did not cross, and retreated > 200 px back
     into the work zone (expires V3 Track 1928 before it grazes the Right line).
  """
  if not state.get("is_locked") or state.get("counted"):
    return False

  state["min_dL_since_lock"] = min(
      state.get("min_dL_since_lock", d_left), d_left
  )
  state["min_dR_since_lock"] = min(
      state.get("min_dR_since_lock", d_right), d_right
  )

  locked_pf = state.get("locked_frame")
  if locked_pf is None:
    return False

  age = current_pf - locked_pf
  if age <= STALE_LOCK_FRAMES:
    return False

  l_pos = state.get("locked_pos")
  if l_pos is not None:
    _, _, lock_dL, lock_dR = l_pos
    prog_l = lock_dL - d_left
    prog_r = lock_dR - d_right
    max_prog = max(prog_l, prog_r)
  else:
    max_prog = 0.0

  min_dL = state.get("min_dL_since_lock", d_left)
  min_dR = state.get("min_dR_since_lock", d_right)

  turned_back_from_left = (
      min_dL <= TURNBACK_NEAR_LINE_PX
      and (d_left - min_dL) >= TURNBACK_RETREAT_PX
  )
  turned_back_from_right = (
      min_dR <= TURNBACK_NEAR_LINE_PX
      and (d_right - min_dR) >= TURNBACK_RETREAT_PX
  )

  wz_empty = state.get("wz_empty_votes", 0)

  if (
      wz_empty >= STALE_LOCK_EMPTY_VOTES and max_prog < STALE_LOCK_TRANSIT_PX
  ) or turned_back_from_left or turned_back_from_right:
    state["is_locked"] = False
    state["locked_class"] = None
    state["locked_frame"] = None
    state["locked_pos"] = None
    state["min_dL_since_lock"] = None
    state["min_dR_since_lock"] = None
    state["predictions"] = [p for p in state["predictions"] if p != "full"]
    state["vote_details"] = [
        v for v in state.get("vote_details", []) if v[0] != "full"
    ]
    return True
  return False


def get_majority_class(state):
  """Resolves the track's majority class ('full', 'empty', or 'Unknown')."""
  if state.get("is_locked") and state.get("locked_class"):
    return state["locked_class"]
  preds = state.get("predictions", [])
  if not preds:
    return "Unknown"
  counts = Counter(preds)
  if counts["full"] > counts["empty"]:
    return "full"
  elif counts["empty"] > counts["full"]:
    return "empty"
  else:
    return preds[-1]


def clear_pre_entry_full_noise(state):
  """When an uncounted track crosses from outside (-1) into the work zone (+1)

  (RETURN_CROSSING_IGNORED), strip any premature 'full' votes/locks acquired
  outside the work zone while preserving 'empty' votes (fixes V3 Track 9
  without losing V3 Tracks 93 & 550 empty history).
  """
  if not state.get("counted"):
    state["is_locked"] = False
    state["locked_class"] = None
    state["locked_frame"] = None
    state["locked_pos"] = None
    state["min_dL_since_lock"] = None
    state["min_dR_since_lock"] = None
    state["predictions"] = [p for p in state["predictions"] if p != "full"]
    state["vote_details"] = [
        v for v in state.get("vote_details", []) if v[0] != "full"
    ]


def reset_track_for_new_count(state, current_pf):
  """Resets a counted track when it returns past +150px inside the work zone."""
  old_cid = state.get("count_id")
  state["counted"] = False
  state["counted_left"] = False
  state["counted_right"] = False
  state["return_armed_left"] = False
  state["return_armed_right"] = False
  state["is_locked"] = False
  state["locked_class"] = None
  state["locked_frame"] = None
  state["locked_pos"] = None
  state["min_dL_since_lock"] = None
  state["min_dR_since_lock"] = None
  state["predictions"] = []
  state["vote_details"] = []
  state["wz_empty_votes"] = 0
  state["cycle_empty_votes"] = 0
  state["last_reset_frame"] = current_pf
  return old_cid


# =====================================================================
# 3. KALMAN CENTROID TRACKER
# =====================================================================
class KalmanCentroidTracker:

  def __init__(
      self, max_disappeared=30, max_distance=180, occlusion_zone_margin=200
  ):
    self.next_object_id = 1
    self.objects = {}
    self.kalman_filters = {}
    self.bboxes = {}
    self.disappeared = {}
    self.velocities = {}
    self.raw_centroids = {}
    self.confidences = {}
    self.max_disappeared = max_disappeared
    self.max_distance = max_distance
    self.occlusion_zone_margin = occlusion_zone_margin

  def _create_kalman(self, cx, cy):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
    )
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
    return kf

  def register(self, centroid, bbox, conf):
    oid = self.next_object_id
    self.next_object_id += 1
    cx, cy = float(centroid[0]), float(centroid[1])
    self.objects[oid] = (int(cx), int(cy))
    self.raw_centroids[oid] = (cx, cy)
    self.bboxes[oid] = bbox
    self.confidences[oid] = float(conf)
    self.disappeared[oid] = 0
    self.velocities[oid] = (0.0, 0.0)
    self.kalman_filters[oid] = self._create_kalman(cx, cy)
    return oid

  def deregister(self, oid):
    for d in (
        self.objects,
        self.raw_centroids,
        self.bboxes,
        self.confidences,
        self.disappeared,
        self.velocities,
        self.kalman_filters,
    ):
      d.pop(oid, None)

  def update(self, rects_with_conf, min_new_track_conf=0.40):
    if len(rects_with_conf) == 0:
      for oid in list(self.disappeared.keys()):
        self.disappeared[oid] += 1
        kf = self.kalman_filters[oid]
        pred = kf.predict()
        px, py = int(pred[0, 0]), int(pred[1, 0])
        self.objects[oid] = (px, py)
        if self.disappeared[oid] > self.max_disappeared:
          self.deregister(oid)
      return self._build_output()

    input_centroids = np.zeros((len(rects_with_conf), 2), dtype=np.float32)
    input_bboxes = []
    input_confs = []
    for i, (x1, y1, x2, y2, conf) in enumerate(rects_with_conf):
      cx = (x1 + x2) / 2.0
      cy = (y1 + y2) / 2.0
      input_centroids[i] = (cx, cy)
      input_bboxes.append((int(x1), int(y1), int(x2), int(y2)))
      input_confs.append(float(conf))

    if len(self.objects) == 0:
      for i in range(len(rects_with_conf)):
        if input_confs[i] >= min_new_track_conf:
          self.register(input_centroids[i], input_bboxes[i], input_confs[i])
      return self._build_output()

    object_ids = list(self.objects.keys())
    predicted_centroids = np.zeros((len(object_ids), 2), dtype=np.float32)
    for idx, oid in enumerate(object_ids):
      pred = self.kalman_filters[oid].predict()
      predicted_centroids[idx] = (pred[0, 0], pred[1, 0])

    D = np.linalg.norm(
        predicted_centroids[:, None, :] - input_centroids[None, :, :], axis=2
    )
    rows = D.min(axis=1).argsort()
    cols = D.argmin(axis=1)[rows]

    used_rows = set()
    used_cols = set()

    for row, col in zip(rows, cols):
      if row in used_rows or col in used_cols:
        continue
      oid = object_ids[row]
      max_d = (
          self.max_distance * 1.35
          if self.disappeared[oid] > 0
          else self.max_distance
      )
      if D[row, col] > max_d:
        continue

      meas = np.array(
          [[input_centroids[col][0]], [input_centroids[col][1]]],
          dtype=np.float32,
      )
      est = self.kalman_filters[oid].correct(meas)
      cx, cy = int(est[0, 0]), int(est[1, 0])
      vx, vy = float(est[2, 0]), float(est[3, 0])

      self.objects[oid] = (cx, cy)
      self.raw_centroids[oid] = (
          float(input_centroids[col][0]),
          float(input_centroids[col][1]),
      )
      self.bboxes[oid] = input_bboxes[col]
      self.confidences[oid] = input_confs[col]
      self.velocities[oid] = (vx, vy)
      self.disappeared[oid] = 0

      used_rows.add(row)
      used_cols.add(col)

    unused_rows = set(range(len(object_ids))) - used_rows
    unused_cols = set(range(len(rects_with_conf))) - used_cols

    for row in unused_rows:
      oid = object_ids[row]
      self.disappeared[oid] += 1
      px, py = int(predicted_centroids[row][0]), int(
          predicted_centroids[row][1]
      )
      self.objects[oid] = (px, py)
      if self.disappeared[oid] > self.max_disappeared:
        self.deregister(oid)

    for col in unused_cols:
      if input_confs[col] >= min_new_track_conf:
        self.register(input_centroids[col], input_bboxes[col], input_confs[col])

    return self._build_output()

  def _build_output(self):
    out = {}
    for oid, (cx, cy) in self.objects.items():
      out[oid] = {
          "centroid": (cx, cy),
          "raw_centroid": self.raw_centroids.get(oid, (float(cx), float(cy))),
          "bbox": self.bboxes[oid],
          "conf": self.confidences.get(oid, 0.0),
          "disappeared": self.disappeared[oid],
          "velocity": self.velocities.get(oid, (0.0, 0.0)),
      }
    return out


# =====================================================================
# 4. REAL-TIME CAPTURE WITH FRAME DROP / SKIP LOGIC
# =====================================================================

class RTSPCapture:
    """Threaded RTSP / stream capture with frame-drop / skip logic.

    Instead of accumulating an unbounded FIFO backlog, the capture thread always
    retains only the newest frame and drops intermediate frames. This eliminates
    backlog delay and memory bloat, guaranteeing real-time low latency.
    """

    def __init__(self, src, timezone, rtsp_transport="tcp", reconnect_delay=2.0,
                 max_retries=0):
        # Convert numeric string like "0" to integer for local webcam
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self.src = src
        self.tz = timezone
        self.rtsp_transport = rtsp_transport
        self.reconnect_delay = float(reconnect_delay)
        self.max_retries = int(max_retries)  # 0 means infinite retries for 24/7 CCTV
        self.is_stream = isinstance(src, int) or (
            isinstance(src, str) and src.startswith(
                ("rtsp://", "rtsps://", "http://", "https://")
            )
        )
        self.cap = None
        self.stop_event = threading.Event()
        self.new_frame_event = threading.Event()
        self.lock = threading.Lock()

        # Latest frame buffer (overwritten continuously; stale frames are dropped)
        self.latest_frame = None
        self.latest_frame_id = 0
        self.latest_captured_at = None
        self.total_captured = 0
        self.dropped_frames = 0
        self.capture_finished = False
        self.thread = None

        if self.is_stream and isinstance(src, str) and src.startswith(("rtsp://", "rtsps://", "http://", "https://")):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self.rtsp_transport}|fflags;nobuffer|flags;low_delay|stimeout;5000000|max_delay;500000"
            )

        connected = self._try_connect()
        if not connected:
            if self.is_stream:
                print(f"WARNING: Initial connection to RTSP stream '{source_label(self.src)}' not established yet.")
                print(f"Starting background reconnection loop (retrying every {self.reconnect_delay:.1f}s)...")
            else:
                raise RuntimeError(f"Could not open video file: {self.src}")

        if self.is_stream:
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

    def _try_connect(self):
        """Attempts to open or reopen the video/stream source cleanly."""
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
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    cap = self.cap
                if cap is None or not cap.isOpened():
                    if self.max_retries > 0 and reconnects >= self.max_retries:
                        print("ERROR: RTSP reconnect retries exhausted.")
                        break
                    reconnects += 1
                    retry_label = f"({reconnects}/{self.max_retries})" if self.max_retries > 0 else f"(attempt {reconnects})"
                    print(f"WARNING: RTSP stream disconnected; reconnecting {retry_label}...")
                    if self.stop_event.wait(self.reconnect_delay):
                        break
                    if self._try_connect():
                        print(f"SUCCESS: Reconnected to RTSP stream '{source_label(self.src)}'.")
                        reconnects = 0
                        failures = 0
                    continue

                try:
                    ret, frame = cap.read()
                except Exception as exc:
                    ret, frame = False, None
                    print(f"WARNING: RTSP read error: {exc}")

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
                        print("WARNING: RTSP frame reads failing; reconnecting...")
                        if not self.stop_event.wait(0.1):
                            if self._try_connect():
                                failures = 0
                                reconnects = 0
                    else:
                        self.stop_event.wait(0.01)
        finally:
            self.capture_finished = True

    def read_next(self, last_fid=None, timeout=0.5):
        """Read the next fresh frame.

        For live streams: returns the latest captured frame, dropping any stale
        frames that accumulated during inference. Waits up to timeout for a new frame.
        For video files: reads sequentially.
        """
        if self.is_stream:
            end_time = time.monotonic() + max(0.0, timeout)
            while not self.stop_event.is_set():
                with self.lock:
                    if self.latest_frame is not None and (last_fid is None or self.latest_frame_id > last_fid):
                        if last_fid is not None and (self.latest_frame_id - last_fid) > 1:
                            self.dropped_frames += (self.latest_frame_id - last_fid - 1)
                        return True, self.latest_frame, self.latest_frame_id, self.latest_captured_at

                if self.capture_finished:
                    return False, None, None, None

                remaining = end_time - time.monotonic()
                if remaining <= 0:
                    break
                self.new_frame_event.wait(timeout=min(0.05, remaining))
                self.new_frame_event.clear()

            with self.lock:
                if self.latest_frame is not None and (last_fid is None or self.latest_frame_id > last_fid):
                    return True, self.latest_frame, self.latest_frame_id, self.latest_captured_at
            return False, None, None, None

        # Video file sequential read
        ret, frame = self.cap.read() if self.cap is not None else (False, None)
        if not ret or frame is None:
            return False, None, None, None
        with self.lock:
            self.total_captured += 1
            fid = self.total_captured
        return True, frame, fid, datetime.datetime.now(self.tz)

    def get_dropped_count(self):
        return self.dropped_frames

    def queue_size(self):
        """Deprecated: always returns 0 since frames are dropped instead of queued."""
        return 0

    def get(self, prop_id):
        if self.cap is not None:
            try:
                return self.cap.get(prop_id)
            except Exception:
                return 0.0
        return 0.0

    def set(self, prop_id, value):
        return self.cap.set(prop_id, value) if self.cap is not None else False

    def stop_capture(self):
        self.stop_event.set()
        self.new_frame_event.set()
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        if self.thread is not None and self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=2.0)
        self.capture_finished = True

    def release(self):
        self.stop_capture()


# =====================================================================
# 5. SINGLE APPEND-ONLY COUNT LOG + QUERY MODE
# =====================================================================

LOG_FIELDS = [
    "event_time",          # Timestamp of the frame associated with the count event
    "logged_at",           # Actual wall-clock time the event was written
    "source_type",         # rtsp or video
    "source",              # Redacted source label (not the RTSP password/URL)
    "session_id",           # Unique process session
    "source_frame",
    "processed_frame",
    "track_id",
    "count_id",
    "line",
    "event",
    "class_name",
    "class_confidence",
    "detector_confidence",
    "added",
    "left_delta",
    "right_delta",
    "total_delta",
    "cumulative_left",
    "cumulative_right",
    "cumulative_total",
    "source_time",         # ISO timestamp for a video if --video-start-datetime is supplied
    "source_seconds",      # Offset into video file; blank for RTSP
]


def now_local(tz):
    return datetime.datetime.now(tz)


def iso_time(dt):
    if dt is None:
        return ""
    if dt.tzinfo is None:
        return dt.isoformat(timespec="milliseconds")
    return dt.isoformat(timespec="milliseconds")


def parse_datetime(value, tz):
    """Parse ISO date/time; naive date/time is interpreted in --timezone."""
    if not value:
        raise ValueError("A timestamp is required.")
    text_value = value.strip().replace("Z", "+00:00")
    dt = datetime.datetime.fromisoformat(text_value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)
    return dt


def source_label(src):
    """Avoid writing stream credentials to the log file."""
    if isinstance(src, str) and src.lower().startswith(("rtsp://", "rtsps://")):
        try:
            from urllib.parse import urlsplit
            p = urlsplit(src)
            return f"{p.scheme}://{p.hostname or 'stream'}:{p.port or ''}{p.path}"
        except Exception:
            return "rtsp_stream"
    return os.path.basename(str(src))


def load_cumulative_from_log(log_path):
    """Restore cumulative counts from the last valid event row after restarts."""
    totals = {"left": 0, "right": 0, "total": 0}
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return totals

    with open(log_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return totals
        missing = set(LOG_FIELDS) - set(reader.fieldnames)
        if missing:
            raise RuntimeError(
                f"Log file schema does not match this script: missing {sorted(missing)}. "
                "Choose a new --log-file or migrate the old CSV first."
            )
        for row in reader:
            try:
                totals = {
                    "left": int(float(row["cumulative_left"])),
                    "right": int(float(row["cumulative_right"])),
                    "total": int(float(row["cumulative_total"])),
                }
            except (TypeError, ValueError, KeyError):
                # Ignore a malformed final row from an abrupt OS-level termination.
                continue
    return totals


class CountEventLogger:
    """Append one durable CSV row per accepted crossing; never rewrites old events."""
    def __init__(self, path):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        needs_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        if not needs_header:
            with open(self.path, "r", newline="", encoding="utf-8") as f:
                existing_header = next(csv.reader(f), [])
            if existing_header != LOG_FIELDS:
                raise RuntimeError(
                    "Existing count log has a different header. Use a new --log-file "
                    "or archive the previous log before starting."
                )
        self.f = open(self.path, "a", newline="", encoding="utf-8", buffering=1)
        self.writer = csv.DictWriter(self.f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if needs_header:
            self.writer.writeheader()
            self.flush_durable()

    def write_count(self, row):
        self.writer.writerow(row)
        self.flush_durable()

    def flush_durable(self):
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        try:
            self.flush_durable()
        finally:
            self.f.close()


def query_log(log_path, query_start, query_end, timezone, time_column="event_time"):
    """Sum count deltas in [query_start, query_end); end is exclusive."""
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"Count log not found: {log_path}")
    if time_column not in {"event_time", "source_time"}:
        raise ValueError("--query-time-column must be event_time or source_time")

    start_dt = parse_datetime(query_start, timezone)
    end_dt = parse_datetime(query_end, timezone)
    if end_dt <= start_dt:
        raise ValueError("Query end must be later than query start.")

    interval = {"left": 0, "right": 0, "total": 0}
    event_rows = 0
    matched_rows = 0

    with open(log_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("Count log is empty or has no header.")
        missing = set(LOG_FIELDS) - set(reader.fieldnames)
        if missing:
            raise RuntimeError(f"Count log missing fields: {sorted(missing)}")

        for row in reader:
            event_rows += 1
            raw_ts = (row.get(time_column) or "").strip()
            if not raw_ts:
                continue
            try:
                ts = parse_datetime(raw_ts, timezone)
            except (ValueError, TypeError):
                continue
            if start_dt <= ts < end_dt:
                matched_rows += 1
                interval["left"] += int(float(row.get("left_delta") or 0))
                interval["right"] += int(float(row.get("right_delta") or 0))
                interval["total"] += int(float(row.get("total_delta") or 0))

    print("\nCOUNT REPORT (end timestamp is exclusive)")
    print("=" * 58)
    print(f"Log file       : {os.path.abspath(log_path)}")
    print(f"Time field     : {time_column}")
    print(f"Start          : {iso_time(start_dt)} (inclusive)")
    print(f"End            : {iso_time(end_dt)} (exclusive)")
    print(f"Count events   : {matched_rows}")
    print(f"Left count     : {interval['left']}")
    print(f"Right count    : {interval['right']}")
    print(f"Total count    : {interval['total']}")
    print("=" * 58)


# Absolute paths relative to the script location for complete portability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DEFAULT_LOG_PATH = os.path.join(SCRIPT_DIR, "production_logs", "counts_log.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="24/7 RF-DETR + ResNet forensic counter. Self-contained and deployable on local PC, server, and cloud."
    )
    parser.add_argument("--source", "-s", default=os.environ.get("SOURCE") or os.environ.get("RTSP_URL"),
                        help="RTSP/HTTP stream URL, webcam ID (e.g. 0), or video file path (defaults to RTSP_URL or SOURCE env var)")
    parser.add_argument("--log-file", default=os.environ.get("LOG_FILE"),
                        help="Append-only count CSV path (defaults to production_logs/counts_log.csv)")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH"),
                        help="ResNet checkpoint path (defaults to models/ResNet152_direction_best_4.pth)")
    parser.add_argument("--detector-checkpoint", default=os.environ.get("DETECTOR_CHECKPOINT"),
                        help="Fine-tuned RF-DETR Small checkpoint path (defaults to models/best_model.pth)")
    parser.add_argument("--mapping", default=os.environ.get("MAPPING_PATH"),
                        help="Class mapping text file (defaults to models/class_mapping.txt)")
    parser.add_argument("--timezone", default=os.environ.get("TIMEZONE", "Asia/Kolkata"),
                        help="Timezone for timestamps, default Asia/Kolkata")
    parser.add_argument("--det-conf", type=float, default=0.20, help="RF-DETR detection threshold (default: 0.20)")
    parser.add_argument("--track-thresh", type=float, default=0.40, help="Minimum confidence to create new tracks (default: 0.40)")
    parser.add_argument("--max-inference-fps", type=float, default=10.0,
                        help="Maximum RF-DETR inference start rate for RTSP; use camera encoder 10 FPS for no backlog (default: 10.0)")
    parser.add_argument("--rtsp-transport", choices=["tcp", "udp"], default="tcp", help="RTSP transport protocol (default: tcp)")
    parser.add_argument("--rtsp-reconnect-delay", type=float, default=2.0, help="Delay in seconds between RTSP reconnect attempts (default: 2.0)")
    parser.add_argument("--rtsp-max-retries", type=int, default=0, help="Max RTSP reconnection attempts (default: 0 for infinite 24/7 retry)")
    parser.add_argument("--video-start-datetime", default=None,
                        help="Optional ISO datetime for start of source video, e.g. 2026-10-27T08:00:00+05:30")
    parser.add_argument("--start", default=None, help="Video start offset in seconds, MM:SS, or HH:MM:SS")
    parser.add_argument("--end", default=None, help="Video end offset in seconds, MM:SS, or HH:MM:SS")
    parser.add_argument("--query-start", default=None, help="Query start datetime, ISO format")
    parser.add_argument("--query-end", default=None, help="Query end datetime, ISO format")
    parser.add_argument("--query-time-column", choices=["event_time", "source_time"], default="event_time",
                        help="Timestamp column for query; use source_time for videos with --video-start-datetime")
    parser.add_argument("--query-only", action="store_true", help="Query existing count log and exit without loading models")
    parser.add_argument("--max-queue-warning", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def ensure_colab_drive_if_needed(path):
    if not path:
        return
    norm = os.path.abspath(path).replace("\\", "/")
    if not norm.startswith("/content/drive/"):
        return
    if os.path.exists("/content/drive/MyDrive"):
        return
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except ImportError:
        pass
    except Exception as exc:
        raise RuntimeError(
            f"Path is configured under Google Drive ({path}) but Drive could not be mounted. "
            "Mount Drive in Colab or use local paths with --log-file, --model and --detector-checkpoint."
        ) from exc


def check_git_lfs_pointer(filepath):
    """Detect if a file is an un-pulled Git LFS pointer text file (~130 bytes)."""
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
    """Intelligently resolves asset/checkpoint paths across:
    1. CLI argument (if provided)
    2. Environment variable (if provided)
    3. Local repo 'models/' directory (<repo_root>/models/)
    4. Local repo root directory (<repo_root>/)
    5. Google Colab Drive paths (if running in Colab)
    """
    if cli_path:
        norm = os.path.abspath(cli_path)
        if os.path.isfile(norm):
            return norm
        raise FileNotFoundError(f"{desc.capitalize()} file specified via CLI argument not found: {cli_path}")

    if env_var and os.environ.get(env_var):
        env_val = os.environ.get(env_var)
        norm = os.path.abspath(env_val)
        if os.path.isfile(norm):
            return norm
        raise FileNotFoundError(f"{desc.capitalize()} file specified via ${env_var} not found: {env_val}")

    search_dirs = [
        MODELS_DIR,
        SCRIPT_DIR,
        os.path.join(SCRIPT_DIR, "..", "models"),
    ]

    for s_dir in search_dirs:
        for fname in candidate_names:
            candidate = os.path.normpath(os.path.join(s_dir, fname))
            if os.path.isfile(candidate):
                return candidate

    # Colab Drive fallbacks
    colab_base = "/content/drive/MyDrive/Amrendra_Guardex/tin_counter_new_angle"
    colab_search_dirs = [
        os.path.join(colab_base, "testing_video_folder_8", "models"),
        os.path.join(colab_base, "model"),
        os.path.join(colab_base, "KD", "runs (1)", "student_train"),
        os.path.join(colab_base, "dataset"),
        "/content/models",
    ]
    for c_dir in colab_search_dirs:
        for fname in candidate_names:
            candidate = os.path.normpath(os.path.join(c_dir, fname))
            if os.path.isfile(candidate):
                return candidate

    return os.path.normpath(os.path.join(MODELS_DIR, candidate_names[0]))


def parse_offset_seconds(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    val_str = str(value).strip()
    if ":" in val_str:
        parts = val_str.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        raise ValueError(f"Invalid offset {value!r}; use seconds, MM:SS, or HH:MM:SS")
    return float(val_str)


def load_class_mapping(mapping_path):
    """Load class mapping from text file with index:name lines."""
    mapping = {}
    if mapping_path and os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    idx, cls_name = line.split(":", 1)
                    try:
                        mapping[int(idx.strip())] = cls_name.strip()
                    except ValueError:
                        continue
    if not mapping:
        mapping = {0: "empty", 1: "full"}
    return mapping


def build_classifier(model_path, num_classes, device):
    """Construct and initialize ResNet152 classifier."""
    resnet = models.resnet152(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    except Exception:
        state_dict = torch.load(model_path, map_location=device)

    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

    resnet.load_state_dict(state_dict)
    resnet = resnet.to(device)
    resnet.eval()
    return resnet


def load_runtime_dependencies():
    """Load CV/ML packages only for inference mode, not for CSV query mode."""
    global cv2, np, Image, RFDETRSmall, torch, nn, models, transforms
    try:
        import cv2 as _cv2
        import numpy as _np
        from PIL import Image as _Image
        from rfdetr import RFDETRSmall as _RFDETRSmall
        import torch as _torch
        import torch.nn as _nn
        from torchvision import models as _models, transforms as _transforms
    except ImportError as exc:
        raise RuntimeError(
            "Inference dependencies are missing. Install them in the runtime first, "
            "for example: pip install -r requirements.txt"
        ) from exc
    cv2, np, Image = _cv2, _np, _Image
    RFDETRSmall, torch, nn = _RFDETRSmall, _torch, _nn
    models, transforms = _models, _transforms


def main():
    args = parse_args()
    try:
        timezone = ZoneInfo(args.timezone)
    except Exception as exc:
        raise RuntimeError(f"Invalid timezone {args.timezone!r}") from exc

    # Intelligently resolve model, checkpoint, mapping, and log paths
    model_path = resolve_asset_path(
        args.model,
        ["ResNet152_direction_best_4.pth", "resnet152.pth"],
        env_var="MODEL_PATH",
        desc="ResNet classifier checkpoint",
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

    default_log = os.path.join(SCRIPT_DIR, "production_logs", "counts_log.csv")
    log_path = os.path.abspath(args.log_file or os.environ.get("LOG_FILE") or default_log)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    ensure_colab_drive_if_needed(log_path)

    # Query mode reads only the CSV. It does not load RF-DETR/ResNet or open a camera.
    if args.query_only or args.query_start or args.query_end:
        if not args.query_start or not args.query_end:
            raise ValueError("Provide both --query-start and --query-end for query mode.")
        query_log(
            log_path,
            args.query_start,
            args.query_end,
            timezone,
            args.query_time_column,
        )
        return

    source = args.source or os.environ.get("SOURCE") or os.environ.get("RTSP_URL")
    if not source:
        raise ValueError(
            "Missing video or stream source.\n"
            "Provide --source <path_or_url> or set RTSP_URL or SOURCE in your environment.\n"
            "Examples:\n"
            "  Local video : python main.py --source sample.mp4\n"
            "  RTSP camera : python main.py --source rtsp://user:pass@192.168.1.50:554/stream1\n"
            "  Webcam      : python main.py --source 0\n"
            "  Query log   : python main.py --query-only --query-start 2026-10-27T08:00:00 --query-end 2026-10-27T18:00:00"
        )

    load_runtime_dependencies()
    ensure_colab_drive_if_needed(model_path)
    ensure_colab_drive_if_needed(detector_checkpoint)

    # Validate model checkpoints exist
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"ResNet checkpoint not found at: {model_path}\n"
            f"Please verify that 'models/ResNet152_direction_best_4.pth' exists or specify --model."
        )
    if not os.path.isfile(detector_checkpoint):
        raise FileNotFoundError(
            f"RF-DETR checkpoint not found at: {detector_checkpoint}\n"
            f"Please verify that 'models/best_model.pth' exists or specify --detector-checkpoint."
        )

    # Check for Git LFS pointer text files
    for name, path in [("ResNet Classifier", model_path), ("RF-DETR Detector", detector_checkpoint)]:
        if check_git_lfs_pointer(path):
            print("\n" + "=" * 75)
            print(f"⚠️  GIT LFS WARNING: {name} weight file is a Git LFS pointer text file:")
            print(f"    {path}")
            print("\nThe actual binary weights (>100MB) have not been downloaded by Git yet.")
            print("To download the actual model files on your machine, run:")
            print("    git lfs install")
            print("    git lfs pull")
            print("=" * 75 + "\n")
            raise RuntimeError(
                f"Model file '{os.path.basename(path)}' is a Git LFS pointer. Run 'git lfs pull' to download weights."
            )

    cumulative = load_cumulative_from_log(log_path)
    session_id = str(uuid.uuid4())
    source_type = "rtsp" if str(source).startswith(("rtsp://", "rtsps://", "http://", "https://")) else "video"
    safe_source_name = source_label(source)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(" 🥫 Guardex AI - Industrial Forensic Counter")
    print("=" * 60)
    print(f"Device               : {device}")
    if device.type == "cpu":
        print("Note                 : Running on CPU. GPU is recommended for full 10 FPS.")
    print(f"Source type          : {source_type}")
    print(f"Source               : {safe_source_name}")
    print(f"Detector Checkpoint  : {detector_checkpoint}")
    print(f"Classifier Model     : {model_path}")
    print(f"Class Mapping        : {mapping_path}")
    print(f"Count Log (CSV)      : {log_path}")
    print(
        f"Restored Cumulative  : left={cumulative['left']}, "
        f"right={cumulative['right']}, total={cumulative['total']}"
    )
    print(f"Session ID           : {session_id}")
    print("=" * 60)

    if source_type == "rtsp":
        print(
            f"Live stream capture: real-time latest-frame sampling with frame-drop at {args.max_inference_fps:.2f} FPS (zero backlog)."
        )

    class_names = load_class_mapping(mapping_path)
    detector = RFDETRSmall.from_checkpoint(detector_checkpoint)
    classifier = build_classifier(model_path, len(class_names), device)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    logger = CountEventLogger(log_path)
    cap = None
    interrupted = False
    processed_frame = 0
    source_frame = 0
    detector_calls = 0
    classifier_calls = 0
    start_wall = time.monotonic()
    next_count_seq = 1
    tracker = KalmanCentroidTracker(max_disappeared=30, max_distance=180)
    track_states = {}
    session_counts = {"left": 0, "right": 0, "total": 0}
    source_anchor = parse_datetime(args.video_start_datetime, timezone) if args.video_start_datetime else None
    last_inference_start = None
    last_processed_fid = None
    video_frames_read_since_start = 0

    def write_count_event(
        event_dt, source_dt, source_seconds, frame_no, processed_no, tid,
        state, side, class_name, class_confidence, detector_confidence, added,
    ):
        # Global counters are restored from the existing append-only log.
        cumulative["total"] += int(added)
        if side == "left":
            cumulative["left"] += int(added)
            session_counts["left"] += int(added)
        else:
            cumulative["right"] += int(added)
            session_counts["right"] += int(added)
        session_counts["total"] += int(added)

        row = {
            "event_time": iso_time(event_dt),
            "logged_at": iso_time(now_local(timezone)),
            "source_type": source_type,
            "source": safe_source_name,
            "session_id": session_id,
            "source_frame": frame_no,
            "processed_frame": processed_no,
            "track_id": tid,
            "count_id": state["count_id"],
            "line": side,
            "event": "COUNTED",
            "class_name": class_name,
            "class_confidence": "" if class_confidence is None else f"{class_confidence:.5f}",
            "detector_confidence": f"{float(detector_confidence):.5f}",
            "added": int(added),
            "left_delta": int(added) if side == "left" else 0,
            "right_delta": int(added) if side == "right" else 0,
            "total_delta": int(added),
            "cumulative_left": cumulative["left"],
            "cumulative_right": cumulative["right"],
            "cumulative_total": cumulative["total"],
            "source_time": iso_time(source_dt),
            "source_seconds": "" if source_seconds is None else f"{source_seconds:.3f}",
        }
        # Persist immediately so a normal interruption does not lose count events.
        logger.write_count(row)
        state["counted_value"] = int(added)
        state["counted"] = True
        state["counted_left"] = side == "left"
        state["counted_right"] = side == "right"
        print(
            f"COUNTED {side.upper()} | +{added} | "
            f"Left={cumulative['left']} Right={cumulative['right']} Total={cumulative['total']} | "
            f"time={iso_time(event_dt)}"
        )

    # Register SIGTERM handler for clean Docker / systemd termination
    try:
        import signal
        def sigterm_handler(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, sigterm_handler)
    except (ValueError, AttributeError, ImportError):
        pass

    try:
        cap = RTSPCapture(
            source,
            timezone=timezone,
            rtsp_transport=args.rtsp_transport,
            reconnect_delay=args.rtsp_reconnect_delay,
            max_retries=args.rtsp_max_retries,
        )
        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        if not reported_fps or not math.isfinite(reported_fps) or reported_fps <= 0 or reported_fps > 120:
            reported_fps = 25.0
        fps = float(reported_fps)

        if source_type == "video":
            start_offset = parse_offset_seconds(args.start) or 0.0
            end_offset = parse_offset_seconds(args.end)
            if start_offset > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_offset * fps))
            start_frame_idx = int(start_offset * fps)
            next_sample_number = 0
            sample_every_native_frame = fps < 10.0
            target_rate = min(10.0, fps)
        else:
            start_offset = 0.0
            end_offset = None
            start_frame_idx = 0
            sample_every_native_frame = True
            target_rate = max(0.1, float(args.max_inference_fps))

        print(f"Input FPS reported: {fps:.3f}; target inference rate: {target_rate:.3f} FPS")
        print("Only the count-event CSV is generated. No annotated video or other report files will be written.")

        while True:
            try:
                if source_type != "video":
                    # Pacing rate-limiter: enforce target FPS without queue lag
                    if last_inference_start is not None:
                        elapsed = time.monotonic() - last_inference_start
                        wait_s = (1.0 / target_rate) - elapsed
                        if wait_s > 0:
                            time.sleep(wait_s)
                    last_inference_start = time.monotonic()

                    ret, frame, fid, captured_at = cap.read_next(last_fid=last_processed_fid, timeout=0.5)
                    if not ret:
                        if cap.capture_finished:
                            break
                        continue
                    last_processed_fid = fid
                    source_frame = int(fid) if fid is not None else source_frame + 1
                    source_seconds = None
                    source_event_dt = captured_at
                    event_dt = captured_at
                else:
                    ret, frame, fid, captured_at = cap.read_next(timeout=0.5)
                    if not ret:
                        break

                    # Reconstruct the actual 1-based source frame index after a seek.
                    video_frames_read_since_start += 1
                    source_frame = start_frame_idx + video_frames_read_since_start
                    source_frame_zero = source_frame - 1
                    relative_frame = video_frames_read_since_start - 1
                    source_seconds = source_frame_zero / max(fps, 1e-6)
                    if end_offset is not None and source_seconds >= end_offset:
                        break
                    if not sample_every_native_frame:
                        target_source_frame = int(round(next_sample_number * fps / target_rate))
                        if relative_frame < target_source_frame:
                            continue
                        next_sample_number += 1
                    if source_anchor is not None:
                        source_event_dt = source_anchor + datetime.timedelta(seconds=source_seconds)
                    else:
                        source_event_dt = None
                    event_dt = now_local(timezone)

                processed_frame += 1
                frame_h, frame_w = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = detector.predict(Image.fromarray(rgb_frame), threshold=float(args.det_conf))
                detector_calls += 1

                rects = []
                if hasattr(detections, "xyxy"):
                    for xyxy, conf, cls_id in zip(
                        detections.xyxy, detections.confidence, detections.class_id
                    ):
                        if int(cls_id) == 0 and float(conf) >= float(args.det_conf):
                            x1, y1, x2, y2 = [float(v) for v in xyxy]
                            rects.append((x1, y1, x2, y2, float(conf)))

                tracked = tracker.update(rects, min_new_track_conf=float(args.track_thresh))

                for tid, info in tracked.items():
                    if info["disappeared"] > 0:
                        continue
                    cx, cy = info["centroid"]
                    raw_cx, raw_cy = info["raw_centroid"]
                    x1, y1, x2, y2 = info["bbox"]
                    det_conf = float(info["conf"])

                    d_left = get_inside_distance((cx, cy), line_pt1, line_pt2, LEFT_INSIDE_SIGN)
                    d_right = get_inside_distance((cx, cy), right_line_pt1, right_line_pt2, RIGHT_INSIDE_SIGN)

                    if tid not in track_states:
                        cid = f"{session_id[:8]}-C{next_count_seq:07d}"
                        next_count_seq += 1
                        init_l = 1 if d_left >= LEFT_RETURN_DEADBAND_PX else (-1 if d_left <= -LEFT_CROSSING_BUFFER_PX else None)
                        init_r = 1 if d_right >= RIGHT_RETURN_DEADBAND_PX else (-1 if d_right <= -RIGHT_CROSSING_BUFFER_PX else None)
                        track_states[tid] = {
                            "count_id": cid,
                            "start_frame": processed_frame,
                            "history": deque(maxlen=30),
                            "predictions": [],
                            "vote_details": [],
                            "wz_empty_votes": 0,
                            "cycle_empty_votes": 0,
                            "is_locked": False,
                            "locked_class": None,
                            "locked_frame": None,
                            "locked_pos": None,
                            "min_dL_since_lock": None,
                            "min_dR_since_lock": None,
                            "counted": False,
                            "counted_left": False,
                            "counted_right": False,
                            "return_armed_left": False,
                            "return_armed_right": False,
                            "side_left": init_l,
                            "side_right": init_r,
                            "seen_work_zone": False,
                            "max_inside_left": d_left,
                            "max_inside_right": d_right,
                            "last_cls_frame": -999,
                            "last_reset_frame": -999,
                            "last_class_conf": None,
                            "last_class_prediction": "Unknown",
                        }

                    state = track_states[tid]
                    state["history"].append((cx, cy))
                    state["max_inside_left"] = max(state["max_inside_left"], d_left)
                    state["max_inside_right"] = max(state["max_inside_right"], d_right)
                    if d_left >= MIN_WORK_ZONE_DEPTH_PX and d_right >= MIN_WORK_ZONE_DEPTH_PX:
                        state["seen_work_zone"] = True

                    direction = None
                    if len(state["history"]) >= 3:
                        dx = state["history"][-1][0] - state["history"][-3][0]
                        if dx <= -4:
                            direction = "to_left"
                        elif dx >= 4:
                            direction = "to_right"

                    check_stale_lock_expiration(state, processed_frame, cx, cy, d_left, d_right)
                    in_reset_cooldown = (processed_frame - state["last_reset_frame"]) <= 5

                    # ResNet inference is only performed on detections in the selected RF-DETR frame.
                    if in_reset_cooldown:
                        pass
                    elif not state["is_locked"]:
                        near_line = abs(d_left) <= 220.0 or abs(d_right) <= 220.0
                        early_track = len(state["predictions"]) < 4
                        throttle_ok = (processed_frame - state["last_cls_frame"]) >= 5
                        if near_line or early_track or throttle_ok:
                            pad_x = int((x2 - x1) * 0.08)
                            pad_y = int((y2 - y1) * 0.08)
                            cx1 = max(0, int(x1) - pad_x)
                            cy1 = max(0, int(y1) - pad_y)
                            cx2 = min(frame_w, int(x2) + pad_x)
                            cy2 = min(frame_h, int(y2) + pad_y)
                            if cx2 > cx1 and cy2 > cy1:
                                crop = rgb_frame[cy1:cy2, cx1:cx2]
                                tensor = preprocess(Image.fromarray(crop)).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    probs = torch.softmax(classifier(tensor), dim=1)[0]
                                    classifier_calls += 1
                                    conf_tensor, idx_tensor = torch.max(probs, dim=0)
                                    raw_class = class_names[int(idx_tensor)].lower()
                                    cls_name = "full" if "full" in raw_class else "empty"
                                    cls_conf = float(conf_tensor.item())
                                state["last_cls_frame"] = processed_frame
                                state["last_class_conf"] = cls_conf
                                state["last_class_prediction"] = cls_name
                                was_locked = state["is_locked"]
                                evaluate_vote_locking(state, cls_name, cls_conf, cx, cy, d_left, d_right)
                                if state["is_locked"] and not was_locked:
                                    state["locked_frame"] = processed_frame

                    new_l, arm_l, events_l = evaluate_buffer_crossing(
                        state["side_left"], d_left, state["counted_left"], state["return_armed_left"],
                        crossing_buffer_px=LEFT_CROSSING_BUFFER_PX,
                        return_deadband_px=LEFT_RETURN_DEADBAND_PX,
                    )
                    state["side_left"], state["return_armed_left"] = new_l, arm_l
                    new_r, arm_r, events_r = evaluate_buffer_crossing(
                        state["side_right"], d_right, state["counted_right"], state["return_armed_right"],
                        crossing_buffer_px=RIGHT_CROSSING_BUFFER_PX,
                        return_deadband_px=RIGHT_RETURN_DEADBAND_PX,
                    )
                    state["side_right"], state["return_armed_right"] = new_r, arm_r

                    for side, evs, d_curr, max_inside in (
                        ("left", events_l, d_left, state["max_inside_left"]),
                        ("right", events_r, d_right, state["max_inside_right"]),
                    ):
                        for ev in evs:
                            if ev == "return":
                                clear_pre_entry_full_noise(state)
                                continue
                            if ev == "return_seen":
                                continue
                            if ev == "return_confirmed":
                                reset_track_for_new_count(state, processed_frame)
                                state["count_id"] = f"{session_id[:8]}-C{next_count_seq:07d}"
                                next_count_seq += 1
                                if side == "left":
                                    state["max_inside_left"] = d_curr
                                else:
                                    state["max_inside_right"] = d_curr
                                continue
                            if ev != "crossed":
                                continue

                            maj_cls = get_majority_class(state)
                            eligible = (
                                not state["counted"]
                                and state["seen_work_zone"]
                                and max_inside >= MIN_WORK_ZONE_DEPTH_PX
                            )
                            if not eligible:
                                continue

                            # Preserve the existing counter's scoring rule: a full load adds 2;
                            # empty/unknown adds 0 but the crossing is still marked counted.
                            added = 2 if maj_cls == "full" else 0
                            state["counted"] = True
                            state["counted_left"] = side == "left"
                            state["counted_right"] = side == "right"

                            event_timestamp = event_dt
                            source_timestamp = source_event_dt
                            write_count_event(
                                event_timestamp,
                                source_timestamp,
                                source_seconds,
                                source_frame,
                                processed_frame,
                                tid,
                                state,
                                side,
                                maj_cls,
                                state.get("last_class_conf"),
                                det_conf,
                                added,
                            )

                if processed_frame % 100 == 0:
                    elapsed = max(time.monotonic() - start_wall, 1e-6)
                    dropped = cap.get_dropped_count() if source_type != "video" else 0
                    print(
                        f"STATUS {iso_time(now_local(timezone))} | frames={processed_frame} | "
                        f"RF-DETR FPS={detector_calls / elapsed:.2f} | dropped={dropped} | "
                        f"left={cumulative['left']} right={cumulative['right']} total={cumulative['total']}"
                    )

            except KeyboardInterrupt:
                interrupted = True
                print("\nINTERRUPTION RECEIVED: stopping capture immediately...")
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nINTERRUPTION RECEIVED: stopping capture. All accepted count events have already been flushed to the CSV log.")
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception as exc:
                print(f"WARNING: error releasing capture: {exc}")
        try:
            logger.close()
        except Exception as exc:
            print(f"WARNING: error closing count log: {exc}")

    elapsed = max(time.monotonic() - start_wall, 1e-6)
    dropped = cap.get_dropped_count() if (cap is not None and source_type != "video") else 0
    print("\n" + "=" * 60)
    print("RUN STOPPED SAFELY" if interrupted else "RUN FINISHED")
    print(f"Count log: {log_path}")
    print(f"Cumulative LEFT: {cumulative['left']}")
    print(f"Cumulative RIGHT: {cumulative['right']}")
    print(f"Cumulative TOTAL: {cumulative['total']}")
    print(f"This session LEFT delta: {session_counts['left']}")
    print(f"This session RIGHT delta: {session_counts['right']}")
    print(f"This session TOTAL delta: {session_counts['total']}")
    print(f"Processed frames: {processed_frame}; RF-DETR calls: {detector_calls}; ResNet crop calls: {classifier_calls}")
    if source_type != "video":
        print(f"Frames dropped to maintain target FPS: {dropped}")
    print(f"Wall-clock detector throughput: {detector_calls / elapsed:.2f} FPS")
    print("Count events are appended and fsynced individually; previous counts are restored from the log on restart.")


if __name__ == "__main__":
    main()
