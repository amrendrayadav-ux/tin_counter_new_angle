# 🥫 Industrial Forensic Tin & Box Counter System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Detector](https://img.shields.io/badge/Detector-RF--DETR%20Small-green.svg)](https://github.com/roboflow/rf-detr)
[![Tracker](https://img.shields.io/badge/Tracker-ByteTrack-orange.svg)](https://github.com/roboflow/supervision)
[![Classifier](https://img.shields.io/badge/Classifier-ResNet152-purple.svg)](https://pytorch.org/vision/stable/models.html)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-brightgreen.svg)](https://git-lfs.github.com/)

A real-time, forensic computer vision pipeline designed for industrial conveyor belts and factory loading bays. The system reliably tracks workers/boxes, classifies carried load states (*empty* vs. *full*), detects crossing through slanted virtual lines, suppresses jitter, and provides full frame-by-frame forensic telemetry.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Important: GitHub 100MB Limit & Git LFS](#-important-github-100mb-limit--git-lfs)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
  - [1. Run on Local Video](#1-run-on-local-video)
  - [2. Run on Live RTSP Stream](#2-run-on-live-rtsp-stream)
  - [3. Run on Google Colab](#3-run-on-google-colab)
- [Command-Line Arguments Reference](#-command-line-arguments-reference)
- [Forensic Outputs & Telemetry](#-forensic-outputs--telemetry)
- [Core Engineering Fixes & Innovations](#-core-engineering-fixes--innovations)
- [Troubleshooting](#-troubleshooting)

---

## 🔍 Overview

Counting carried objects across industrial factory corridors requires solving severe real-world challenges:
1. **Steep Slanted Crossing Lines**: Traditional horizontal `dx` direction gating fails when workers move diagonally.
2. **Detector Noise & ID Flicker**: Standard DETR outputs without NMS cause tracking jitter and ID fragmentation.
3. **Load State Classification**: Workers may return empty or with 2 tins/boxes; state must be locked robustly using deep visual classification.
4. **Network Jitter on RTSP Streams**: Packet loss and latency lag can starve object tracking.

This project combines:
- **Fine-Tuned RF-DETR Small**: Fast real-time transformer object detector trained via Knowledge Distillation from an RF-DETR 2XL teacher.
- **ByteTrack Tracking**: Associate low- and high-score detection boxes for unbroken track continuity.
- **One-Euro Adaptive Smoothing Filter**: Dynamically smooths bounding boxes (heavy filtering when still, low latency when moving fast).
- **ResNet-152 Classifier**: Classifies cropped worker/load regions into `empty` (value 0) or `full` (value 2).
- **Hysteresis Boundary Gates**: Inside-to-outside work area crossing logic eliminates boundary oscillation.

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    A[Video / RTSP Camera Stream] --> B[RTSP Low-Latency Capture / Decoupled Buffer]
    B --> C[RF-DETR Small Detection]
    C --> D[Class-Agnostic NMS]
    D --> E[ByteTrack Multi-Object Tracker]
    E --> F[One-Euro Jitter Filter]
    F --> G[ResNet152 Crop Classifier\nempty: 0 | full: 2]
    G --> H{LineGate Hysteresis\nLeft & Right Lines}
    H -->|Inside -> Outside| I[Count Increment & Deduplication]
    I --> J[Forensic Video, CSV Telemetry & Trajectory Maps]
```

---

## 📁 Repository Structure

```
tin_counter_final/
├── models/
│   ├── ResNet152_direction_best_4.pth  # Trained ResNet152 classification weights (~223 MB)
│   ├── best_model.pth                  # Fine-tuned RF-DETR Small checkpoint (~121 MB)
│   ├── classifier_mapping.txt          # Class index mapping (0:empty, 1:full)
│   └── class_mapping.txt               # Fallback mapping file
├── main.py                             # Main production entry point with intelligent local path resolver
├── rfdetr_counter_return_buffer.py     # Standalone drop-in script with all 13 forensic fixes
├── colab_quickstart.ipynb              # Ready-to-run Google Colab Notebook
├── requirements.txt                    # Pinned Python package dependencies
├── .gitattributes                      # Pre-configured Git LFS tracking for *.pth and *.pt models
├── .gitignore                          # Standard git ignore for outputs, videos, and virtual environments
├── setup_github.bat                    # One-click Windows setup script for Git LFS & repo commit
├── setup_github.sh                     # One-click Linux / Mac / Colab setup script
└── README.md                           # Documentation
```

---

## ⚠️ Important: GitHub 100MB Limit & Git LFS

GitHub rejects any file larger than **100 MB** on standard `git push`.
In this repository:
- `models/ResNet152_direction_best_4.pth` is **~222.7 MB**
- `models/best_model.pth` is **~121.6 MB**

Both files are tracked using **Git LFS (Large File Storage)** via `.gitattributes`.

### Setting up Git LFS before pushing to GitHub:

1. **Install Git LFS**:
   - **Windows**: Download from [git-lfs.github.com](https://git-lfs.github.com/) or run `winget install GitHub.GitLFS`
   - **Ubuntu/Debian**: `sudo apt-get install git-lfs`
   - **macOS**: `brew install git-lfs`

2. **Initialize Git & Push**:
   Simply run the included helper script:
   ```bash
   # On Windows:
   setup_github.bat

   # On Linux/macOS:
   chmod +x setup_github.sh
   ./setup_github.sh
   ```

   Or manually execute:
   ```bash
   git lfs install
   git init
   git branch -M main
   git lfs track "*.pth" "*.pt" "*.onnx"
   git add .gitattributes
   git add .
   git commit -m "Initial commit: Forensic Tin Counter with RF-DETR Small and ResNet152"
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

> **Alternative (GitHub Releases / Cloud Download)**: If you do not want to use Git LFS bandwidth on GitHub, you can add `models/*.pth` to `.gitignore`, upload the weights to **GitHub Releases** or **Google Drive**, and download them when deploying.

---

## 💻 Installation

Ensure Python 3.10 or higher is installed with CUDA-enabled PyTorch.

```bash
# Clone your repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# Create a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note on Supervision**: `supervision` is specifically pinned to `>=0.25,<0.31` in `requirements.txt` because `sv.ByteTrack` was removed starting in `supervision 0.31`.

---

## 🚀 Quick Start

The script automatically detects and resolves default paths to `./models/` if CLI arguments are omitted.

### 1. Run on Live RTSP Stream (24/7 Production Mode)

```bash
python main.py \
  --source "rtsp://admin:password@192.168.1.100:554/stream1" \
  --transport tcp \
  --fps 10.0
```

### 2. Run on Local Video or Webcam

```bash
# Webcam 0:
python main.py --source 0

# Local Video file:
python main.py --source "path/to/factory_recording.mp4"
```

---

## ⚙️ Command-Line Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source`, `-s` | `str` | *Environment / Arg* | RTSP camera stream URL (`rtsp://...`), webcam index (`0`), or video file |
| `--log-file` | `str` | `production_logs/counts_log.csv` | Append-only CSV ledger of increasing counts in IST Day and Time |
| `--timing-log` | `str` | `production_logs/pipeline_timing.csv` | Per-frame pipeline execution timing CSV (model, classifier, pipeline ms) |
| `--no-timing-log` | `flag` | `False` | Disables writing the per-frame timing log |
| `--fps` | `float` | `10.0` | Target processing FPS rate limiter pacing |
| `--timezone` | `str` | `Asia/Kolkata` | Timezone for timestamps (IST) |
| `--model` | `str` | `models/ResNet152_direction_best_4.pth` | Path to ResNet152 classifier weights |
| `--detector-checkpoint` | `str` | `models/best_model.pth` | Path to fine-tuned RF-DETR Small checkpoint |
| `--mapping` | `str` | `models/class_mapping.txt` | Path to class mapping file (`0:empty`, `1:full`) |
| `--det-conf` | `float` | `0.20` | RF-DETR detection confidence floor |
| `--track-thresh` | `float` | `0.40` | ByteTrack track activation threshold |
| `--transport` | `str` | `tcp` | RTSP network transport protocol (`tcp` or `udp`) |
| `--reconnect-delay` | `float` | `2.0` | RTSP reconnect backoff delay in seconds |
| `--max-retries` | `int` | `0` | Max reconnection retries (`0` for infinite 24/7 auto-retry) |
| `--realtime` | `flag` | `False` | Forces real-time frame drop and rate limiter on video files |
| `--show` | `flag` | `False` | Displays real-time OpenCV window (optional preview) |

---

## 📊 Telemetry & Production Logs

The counter operates in a silent, high-performance mode: **all video encoding, trajectory maps, and terminal spam are removed**. The pipeline produces only two durable logs:

### 1. Pure Increasing Count Log (`production_logs/counts_log.csv`)
Records an append-only row strictly when a worker crosses with a positive load:
- **`ist_day`**: Day of week (e.g. `Wednesday`)
- **`ist_time`**: Timestamp in IST (e.g. `06:45:12.345`)
- **`ist_date`**: Date in IST (e.g. `2026-10-28`)
- **`ist_datetime`**: Combined IST timestamp
- **`count_increment`**: Count delta added (`+2`)
- **`line`**: `left` or `right` line
- **`class_name`**: `full`
- **`cumulative_total`**, **`cumulative_left`**, **`cumulative_right`**: Persistent running totals
- **`track_id`**, **`count_id`**, **`frame_number`**, **`source`**

**Console Output:**
```text
[Wednesday, 2026-10-28 06:45:12 IST] COUNT INCREASED: +2 | Total=42 (Left=22, Right=20) | Line=LEFT | Class=full
```

### 2. Per-Frame Execution Timing Log (`production_logs/pipeline_timing.csv`)
Logs latency telemetry for every single processed frame:
- **`ist_day`**: Day of the week in IST
- **`ist_time`**: IST Timestamp
- **`frame_number`**: Frame sequence number
- **`model_time_ms`**: Time taken by RF-DETR Small object detector (ms)
- **`classifier_time_ms`**: Time taken by ResNet152 crop classifier (ms, `0.00` if no crops)
- **`pipeline_time_ms`**: Total end-to-end execution time for the frame (ms)
- **`pipeline_fps`**: Instantaneous processing FPS (`1000 / pipeline_time_ms`)
- **`num_detections`**: Number of detections in frame
- **`num_crops_classified`**: Number of tin crops classified in frame

---

## 🔬 Core Engineering Innovations

1. **24/7 RTSP Real-Time Frame-Drop**: The capture thread retains only the newest frame and drops older frames during inference, keeping latency strictly at 0 ms with infinite reconnection recovery.
2. **Kinematic Loitering Gate**: Loitering workers (`wz_empty_votes >= 10`) must achieve active motion ($\ge 120\text{ px}$ displacement) and $\ge 95\%$ mean confidence across 4 votes to lock, eliminating stationary false positives.
3. **Stale Lock Expiration**: Mid-room occlusion locks expire if uncrossed within 35 frames.
4. **Shifted Physical Right Exit Line**: Shifted $73\text{ px}$ outward (`RIGHT_COMMIT_PX = 85`) to clear the top-center exit doorway.
5. **Confirmed Return Buffer**: Requires workers to re-enter $\ge 150\text{ px}$ inside the work zone for $0.8\text{ s}$ before arming a new counting cycle.
6. **Cumulative Restoration**: Automatically restores previous session totals on reboot.

---

## 🛠️ Troubleshooting

- **`AttributeError: module 'supervision' has no attribute 'ByteTrack'`**:
  `supervision>=0.31` removed `sv.ByteTrack`. Run:
  ```bash
  pip install "supervision>=0.25,<0.31"
  ```
- **RTSP Stream Lag / Visual Smearing**:
  Use TCP transport to avoid UDP packet loss:
  ```bash
  python main.py --source "rtsp://..." --transport tcp
  ```
- **CUDA Out of Memory**:
  Reduce frame rate via `--target-fps 5` or run on smaller resolutions.

---

## 📄 License

This project is licensed under the Apache 2.0 License.
