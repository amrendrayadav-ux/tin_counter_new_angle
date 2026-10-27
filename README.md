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

The script automatically detects and resolves default paths to `./models/` if CLI arguments are omitted!

### 1. Run on Local Video

```bash
python main.py \
  --source "path/to/factory_recording.mp4" \
  --output-dir "./output" \
  --prefix "forensic_run_1"
```

### 2. Run on Live RTSP Stream

```bash
python main.py \
  --source "rtsp://admin:password@192.168.1.100:554/stream1" \
  --output-dir "./output_rtsp" \
  --prefix "live_cam_1" \
  --transport tcp \
  --show
```
*Press `q` in the OpenCV window to exit gracefully.*

### 3. Run on Google Colab

Open [`colab_quickstart.ipynb`](colab_quickstart.ipynb) directly in Google Colab, select a **T4 GPU** runtime, and run the cells.

---

## ⚙️ Command-Line Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source`, `-s` | `str` | *Interactive Prompt* | Input video file path or RTSP stream URL (`rtsp://...`) |
| `--output-dir`, `-o` | `str` | `./output` | Destination folder for forensic video, CSVs, and telemetry |
| `--prefix` | `str` | `forensic_run_cam3` | Prefix attached to all output files |
| `--model` | `str` | `./models/ResNet152_direction_best_4.pth` | Path to ResNet152 classifier weights |
| `--detector-checkpoint` | `str` | `./models/best_model.pth` | Path to fine-tuned RF-DETR Small checkpoint |
| `--mapping` | `str` | `./models/classifier_mapping.txt` | Path to class mapping file (`0:empty`, `1:full`) |
| `--person-class-id` | `int` | `0` | Target class index in fine-tuned RF-DETR (default `0`) |
| `--det-conf` | `float` | `0.20` | Detection confidence cutoff before tracker filtering |
| `--track-thresh` | `float` | `0.40` | ByteTrack track activation threshold |
| `--smooth-cutoff` | `float` | `0.6` | One-Euro filter cutoff (lower = heavier smoothing when still) |
| `--smooth-beta` | `float` | `0.01` | One-Euro responsiveness (higher = lower lag when fast) |
| `--hyst` | `float` | `12.0` | Line crossing hysteresis dead-band in pixels |
| `--anchor` | `str` | `center` | Tracking anchor position: `center` or `bottom` |
| `--no-stitch` | `flag` | `False` | Disables conservative ID stitching on brief tracker drops |
| `--show` | `flag` | `False` | Displays real-time OpenCV rendering window |
| `--transport` | `str` | `tcp` | RTSP network transport protocol (`tcp` or `udp`) |
| `--duration` | `str` | `None` | Max duration for RTSP stream (e.g. `05:00` or `300`) |
| `--start` | `str` | `None` | Video start timestamp (e.g. `01:30` or seconds) |
| `--end` | `str` | `None` | Video end timestamp (e.g. `10:00` or seconds) |

---

## 📊 Forensic Outputs & Telemetry

Every run produces an audit-ready forensic bundle:

1. **Annotated Video (`<prefix>_video.mp4`)**:
   - Rendered with smoothed bounding boxes, track IDs, motion trajectories, crossing lines with hysteresis zones, and live telemetry HUD.
2. **Summary CSV (`<prefix>_summary.csv`)**:
   - Rollup of every track: `Track_ID`, `Start_Frame`, `End_Frame`, `Final_Class`, `Counted_Value` (0 or 2), `Reason`, `ID_Chain`.
3. **Per-Frame Details CSV (`<prefix>_per_frame_details.csv`)**:
   - Complete record of raw vs. smoothed coordinates, confidence, bounding boxes, ResNet raw prediction, and lock status for each frame.
4. **Count Events CSV (`<prefix>_count_events.csv`)**:
   - Chronological ledger detailing the exact frame and geometric conditions under which every count increment occurred.
5. **Telemetry Dump JSON (`<prefix>_telemetry_dump.json`)**:
   - Complete machine-readable JSON containing full trajectories and run metadata.
6. **Trajectory Map (`<prefix>_trajectory_map.jpg`)**:
   - Bird's-eye composite visualization overlaying all trajectories onto the background frame.

---

## 🔬 Core Engineering Fixes & Innovations

1. **RGB Color Alignment**: Fixed RGB vs. BGR frame formatting for RF-DETR Transformer inputs.
2. **One-Euro Adaptive Filtering**: Eliminated jitter on stationary workers without introducing tracking lag during rapid motion.
3. **Hysteresis Crossing Gate**: Eliminated boundary oscillation and double counts using dead-band buffers on slanted lines.
4. **Geometry-Based Crossing (Inside $\to$ Outside)**: Computes dot-product direction from an inside work-area reference, avoiding failures of naive horizontal $dx$ gating on slanted boundaries.
5. **Batch ResNet Inference**: Crops from all active tracks in a frame are batched together for GPU acceleration.
6. **Conservative ID Stitching**: Prevents count duplication when a tracker briefly drops and assigns a new ID within spatial proximity.
7. **Threaded Low-Latency RTSP Capture**: Background thread continuously flushes stale frames, ensuring real-time inference on slow hardware without buffering lag.

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
