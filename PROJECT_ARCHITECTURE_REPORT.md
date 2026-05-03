# Driver Drowsiness AI - Complete Project Architecture Report

Generated on: April 15, 2026

## 1) Project Purpose

This project is a real-time driver fatigue and distraction monitoring system.
It detects unsafe behavior from a live camera stream, escalates alerts, and can trigger emergency intervention UI.

Core goals:
- Detect drowsiness early and reliably
- Reduce false positives with temporal smoothing and hysteresis
- Provide clear visual/audio interventions
- Support future ML experimentation and model upgrades

## 2) Two Runtime Paths in This Repository

### A. Modern Runtime (Primary)
- app.py
- detection_engine.py

This is the main architecture currently intended for full UX and robust detection flow.

### B. Legacy Runtime (Fallback / Older Path)
- driver_alertness.py

This script uses classic OpenCV + Haar cascades + optional pickle model and is useful as an older baseline.

## 3) High-Level System Architecture

### UI / Orchestration Layer
- app.py contains:
  - Startup splash animation
  - Main dashboard window
  - Full-screen critical alert window

### Detection Layer
- detection_engine.py contains:
  - MediaPipe FaceLandmarker inference
  - EAR, MAR, PERCLOS, head-pose computation
  - State machine: NORMAL -> WARNING -> CRITICAL

### Threading / Data Flow
- A detection thread captures camera frames and calls DetectionEngine.process_frame(frame)
- It emits frame + DetectionResult to dashboard UI via Qt signals
- UI updates gauges/cards/banner and triggers critical overlay once per critical transition

### Intervention Layer
- Full-screen red alert view
- Audio alarm sequence (beep + optional TTS)
- 10-second auto-call countdown using tel: URI

## 4) End-to-End Real-Time Workflow

1. App starts and shows startup splash.
2. Dashboard starts detection thread.
3. Camera frame captured with OpenCV.
4. Frame sent to MediaPipe FaceLandmarker.
5. Landmarks converted to metrics:
   - EAR (Eye Aspect Ratio)
   - MAR (Mouth Aspect Ratio)
   - PERCLOS (percent of eye closure over window)
   - Head pose (yaw, pitch, roll via solvePnP)
6. Unsafe indicators are fused into a confidence and reason string.
7. Stateful temporal logic decides NORMAL/WARNING/CRITICAL.
8. Dashboard renders:
   - Camera overlay
   - Gauges/cards
   - Safety score
   - Alert banner
9. On first CRITICAL transition:
   - Full-screen alert opens
   - Audio intervention starts
   - Countdown starts
   - Optional emergency call launch
10. If user dismisses and recovery holds, state returns to safe mode.

## 5) Detection Logic and Model Details

### Landmark Model at Runtime
- models/face_landmarker.task
- Used by detection_engine.py via MediaPipe Tasks API

### Core Metrics
- EAR:
  EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

- MAR:
  MAR = ||top-bottom|| / ||left-right||

- PERCLOS:
  PERCLOS = closed_eye_frames / total_frames_in_window

### Default Thresholds and Timing (DetectionEngine)
- ear_threshold = 0.21
- mar_threshold = 0.65
- perclos_threshold = 0.40
- warning_delay_sec = 5.0
- critical_delay_sec = 7.0
- recovery_sec = 3.0

### Why This Works
- Uses physiological + behavioral cues together
- Uses temporal persistence (not single-frame spikes)
- Uses hysteresis to reduce alert flicker

## 6) State Machine Behavior

States:
- NORMAL
- WARNING
- CRITICAL

Transition pattern:
- Sustained unsafe behavior -> WARNING
- Longer sustained unsafe behavior -> CRITICAL
- Sustained safe behavior -> recovery to NORMAL

Unsafe signals include:
- Eyes closed
- High PERCLOS
- Yawning
- Head turned
- Nodding

## 7) Dashboard and UI Outputs

Live UI outputs:
- EAR gauge
- MAR gauge
- PERCLOS card
- Yawn status
- Head-pose status
- Danger confidence percentage
- Safety score grade
- Alert counter
- Footer status and FPS

Critical intervention outputs:
- Full-screen emergency overlay
- Audible alarm and spoken warning (if TTS available)
- 10-second auto-call countdown
- Manual controls: awake, mute, call

## 8) Training Pipeline (Classical ML Track)

Dataset:
- assets/dataset/dataset.csv

Training scripts:
- train_model.py (basic baseline)
- train_model_99.py (advanced ensemble)
- train_model_advanced.py (another optimized ensemble path)
- train_model_deep_learning.py (experimental DL path)

Saved artifacts in models/:
- driver_alertness_model.pkl
- label_encoder.pkl
- scaler.pkl
- all_models_info.pkl

Important note:
- Modern app.py runtime does NOT depend on these pickle classifiers.
- app.py primarily uses MediaPipe landmarks + rule/state logic.
- driver_alertness.py is the script that uses pickle-based classifier predictions.

## 9) Dataset and Features

CSV columns include:
- category
- image_id
- filename
- class_label
- behavior_description
- severity_level
- bounding_box_x, bounding_box_y, bounding_box_w, bounding_box_h
- timestamp

Classical training derives tabular features from metadata such as:
- severity
- bbox geometry
- interaction features
- hashed text-derived proxies

## 10) Dependencies and Environment Strategy

Runtime requirements (requirements.txt):
- opencv-python
- numpy
- scipy
- scikit-learn
- pandas
- joblib
- mediapipe
- PyQt5

TensorFlow note:
- Not required for primary runtime
- Only needed for train_model_deep_learning.py
- Recommended in separate training environment

## 11) Known Issue and Mitigation

Observed issue (historical logs):
- TensorFlow native DLL load failure on Windows causing MediaPipe optional import path crash

Mitigation already in code:
- detection_engine.py uses a safe MediaPipe import helper that temporarily stubs tensorflow.tools.docs during import

Effect:
- Keeps runtime stable even if TensorFlow install is broken, as long as TensorFlow is not required for active runtime.

## 12) Folder/Asset Overview

Key files:
- app.py (primary app)
- detection_engine.py (primary detection logic)
- driver_alertness.py (legacy runtime)
- requirements.txt
- assets/alert.wav
- assets/wolf_logo.png
- models/face_landmarker.task

Prototype UI assets (not wired to runtime):
- stitch_ai_driver_fatigue_detection_dashboard/.../code.html

Current non-code outputs:
- Logs/ is currently empty
- err.txt, stderr_log.txt, error_lines.json contain previous failure traces

## 13) Practical Run Modes

### Run primary system
python app.py

### Run legacy script
python driver_alertness.py

### Retrain classical ensemble
python train_model_99.py

### Optional deep-learning experiment
python train_model_deep_learning.py

## 14) Strengths and Gaps

Strengths:
- Strong real-time UI and alerting flow
- Multi-signal fatigue detection
- Temporal smoothing + robust state machine
- Good separation between runtime detection and training experiments

Gaps:
- Classical models are metadata-driven (limited ceiling)
- Deep-learning script is currently synthetic-feature based, not full real image pipeline
- Production-grade logging and deployment packaging can be expanded

## 15) Suggested Next Engineering Steps

1. Consolidate one official runtime path (app.py) and mark legacy script clearly.
2. Add structured logging (event timestamps, metric snapshots, alert reasons).
3. Add config file for thresholds/timers rather than hardcoded constants.
4. Build evaluation harness with recorded video clips and ground truth labels.
5. Migrate DL path to true image-based training and robust validation splits.
6. Add deployment profile (CPU-only runtime env and separate training env).

## 16) Final Summary

This repository already contains a capable real-time fatigue intervention system.
The most important architecture is:
- app.py for UX and orchestration
- detection_engine.py for landmark-driven temporal detection

The training scripts and pickled models are valuable experimentation assets, but they are secondary to the modern real-time landmark pipeline used by the primary application.

That distinction is key for maintenance, scaling, and production hardening.
