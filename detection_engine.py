"""
Detection Engine — MediaPipe FaceLandmarker-based driver drowsiness detection.

Uses the new MediaPipe Tasks API (0.10.x+) with FaceLandmarker for:
  - Eye Aspect Ratio (EAR) with temporal smoothing
  - PERCLOS (% eye closure over sliding window)
  - Mouth Aspect Ratio (MAR) for yawn detection
  - Head pose estimation (pitch/yaw/roll) via solvePnP
  - State machine: NORMAL → WARNING → CRITICAL with hysteresis
"""

import cv2
import numpy as np
import time
import os
import sys
import types
import importlib
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from scipy.spatial import distance as dist


def _install_tensorflow_doc_stubs():
    """Install minimal TensorFlow stubs so MediaPipe can import cleanly."""
    injected_modules = []

    tensorflow_module = types.ModuleType("tensorflow")
    tensorflow_module.__path__ = []

    tensorflow_tools_module = types.ModuleType("tensorflow.tools")
    tensorflow_tools_module.__path__ = []

    tensorflow_docs_module = types.ModuleType("tensorflow.tools.docs")

    def _no_op_decorator(obj):
        return obj

    tensorflow_docs_module.doc_controls = types.SimpleNamespace(
        do_not_generate_docs=_no_op_decorator
    )

    sys.modules["tensorflow"] = tensorflow_module
    injected_modules.append("tensorflow")
    sys.modules["tensorflow.tools"] = tensorflow_tools_module
    injected_modules.append("tensorflow.tools")
    sys.modules["tensorflow.tools.docs"] = tensorflow_docs_module
    injected_modules.append("tensorflow.tools.docs")
    return injected_modules

def _import_mediapipe_tasks_safely():
    """
    Import MediaPipe Tasks with a fallback for optional TensorFlow doc controls.

    MediaPipe's tasks package imports `tensorflow.tools.docs.doc_controls` for
    decorators. If TensorFlow is installed but its native runtime fails to load
    on this machine, that optional import can crash startup. We inject a tiny
    temporary stub only for import-time and then clean it up.
    """
    injected_modules = _install_tensorflow_doc_stubs()
    try:
        mp_module = importlib.import_module("mediapipe")
        base_options = importlib.import_module("mediapipe.tasks.python").BaseOptions
        vision_module = importlib.import_module("mediapipe.tasks.python.vision")
        return (
            mp_module,
            base_options,
            vision_module.FaceLandmarker,
            vision_module.FaceLandmarkerOptions,
            vision_module.RunningMode,
        )
    except ImportError as exc:
        err_text = str(exc)
        if "_pywrap_tensorflow_internal" not in err_text and "tensorflow" not in err_text:
            raise

        # Remove partially-loaded TensorFlow modules before trying one more time.
        for module_name in list(sys.modules.keys()):
            if module_name == "tensorflow" or module_name.startswith("tensorflow."):
                sys.modules.pop(module_name, None)
            if module_name == "mediapipe" or module_name.startswith("mediapipe."):
                sys.modules.pop(module_name, None)

        injected_modules = _install_tensorflow_doc_stubs()
        try:
            mp_module = importlib.import_module("mediapipe")
            base_options = importlib.import_module("mediapipe.tasks.python").BaseOptions
            vision_module = importlib.import_module("mediapipe.tasks.python.vision")
            return (
                mp_module,
                base_options,
                vision_module.FaceLandmarker,
                vision_module.FaceLandmarkerOptions,
                vision_module.RunningMode,
            )
        finally:
            for module_name in reversed(injected_modules):
                sys.modules.pop(module_name, None)


mp, BaseOptions, FaceLandmarker, FaceLandmarkerOptions, RunningMode = _import_mediapipe_tasks_safely()


# ── MediaPipe FaceMesh landmark indices ──────────────────────────────

# Left eye: 6 points (matching EAR formula)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Right eye: 6 points
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Mouth inner landmarks for MAR
MOUTH_INNER_TOP = 13
MOUTH_INNER_BOTTOM = 14
MOUTH_INNER_LEFT = 78
MOUTH_INNER_RIGHT = 308
# Nose tip, chin, left/right eye corner, left/right mouth corner — for solvePnP
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]


class DriverState(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class DetectionResult:
    """Single-frame detection output."""
    state: DriverState = DriverState.NORMAL
    ear: float = 0.0           # smoothed EAR
    raw_ear: float = 0.0       # raw EAR (no smoothing)
    mar: float = 0.0           # smoothed MAR
    perclos: float = 0.0       # % of eye closure in window
    yaw: float = 0.0           # head yaw in degrees
    pitch: float = 0.0         # head pitch in degrees
    roll: float = 0.0          # head roll in degrees
    face_detected: bool = False
    eyes_closed: bool = False
    is_yawning: bool = False
    is_head_turned: bool = False
    is_nodding: bool = False
    confidence: float = 0.0
    alert_reason: str = ""
    closed_duration: float = 0.0  # seconds eyes have been closed
    face_bbox: tuple = ()      # (x, y, w, h) for UI overlay
    landmarks_2d: list = field(default_factory=list)  # for mesh drawing


class DetectionEngine:
    """
    Real-time driver drowsiness detection engine.

    Uses MediaPipe FaceLandmarker (478 landmarks) for:
      - EAR computation with exponential moving average smoothing
      - PERCLOS over configurable sliding window
      - MAR-based yawn detection
      - Head pose via cv2.solvePnP

    State machine with hysteresis prevents rapid flickering.
    """

    def __init__(
        self,
        ear_threshold: float = 0.21,
        mar_threshold: float = 0.65,
        perclos_threshold: float = 0.40,
        head_yaw_threshold: float = 40.0,   # degrees — raised for fewer false positives
        head_pitch_threshold: float = 35.0,  # degrees — raised for fewer false positives
        warning_delay_sec: float = 5.0,      # seconds of sustained unsafe before WARNING
        critical_delay_sec: float = 7.0,     # seconds of sustained unsafe before CRITICAL
        recovery_sec: float = 3.0,           # seconds of all-clear to recover
        ema_alpha: float = 0.3,              # smoothing factor (0=smooth, 1=raw)
        perclos_window_sec: float = 60.0,
        fps_estimate: float = 30.0,
    ):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.perclos_threshold = perclos_threshold
        self.head_yaw_threshold = head_yaw_threshold
        self.head_pitch_threshold = head_pitch_threshold
        self.warning_delay_sec = warning_delay_sec
        self.critical_delay_sec = critical_delay_sec
        self.recovery_sec = recovery_sec
        self.ema_alpha = ema_alpha
        self.perclos_window_sec = perclos_window_sec

        # ── FaceLandmarker (new Tasks API) ──────────────────────────
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models", "face_landmarker.task"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face landmarker model not found at {model_path}. "
                "Download from https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = FaceLandmarker.create_from_options(options)

        # ── ML Model Integration ────────────────────────────────────
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self._load_ml_model()

        # State
        self._state = DriverState.NORMAL
        self._smoothed_ear = 0.30
        self._smoothed_mar = 0.10
        self._closed_counter = 0    # consecutive frames with eyes closed
        self._open_counter = 0      # consecutive frames with eyes open (recovery)
        self._yawn_counter = 0
        self._head_turn_counter = 0
        self._nod_counter = 0

        # Time-based sustained detection
        self._unsafe_start_time = None   # when unsafe behaviour first began
        self._safe_start_time = None     # when safe behaviour first began (for recovery)

        # PERCLOS sliding window (stores 1=closed, 0=open per frame)
        window_size = int(perclos_window_sec * fps_estimate)
        self._perclos_window = deque(maxlen=max(window_size, 30))

        # Timing for closed-duration
        self._eyes_closed_start = None

        # 3D model points for solvePnP (generic face model)
        self._model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye left corner
            (225.0, 170.0, -135.0),    # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0),   # Right mouth corner
        ], dtype=np.float64)

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """Process a single BGR frame and return detection result."""
        result = DetectionResult()
        h, w, _ = frame.shape

        # Convert BGR to RGB and create MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Run face landmarker
        mp_result = self.face_landmarker.detect(mp_image)

        if not mp_result.face_landmarks:
            result.face_detected = False
            # No face → decay counters slowly
            self._closed_counter = max(0, self._closed_counter - 1)
            self._open_counter += 1
            self._perclos_window.append(0)
            self._update_state(result)
            return result

        face_lm = mp_result.face_landmarks[0]
        result.face_detected = True

        # Extract all landmarks as pixel coordinates
        lm = []
        for landmark in face_lm:
            lm.append((int(landmark.x * w), int(landmark.y * h)))
        result.landmarks_2d = lm

        # Compute bounding box from landmarks
        xs = [p[0] for p in lm]
        ys = [p[1] for p in lm]
        x_min, x_max = max(0, min(xs)), min(w, max(xs))
        y_min, y_max = max(0, min(ys)), min(h, max(ys))
        result.face_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        # ── EAR ──────────────────────────────────────────────────────
        left_ear = self._compute_ear(lm, LEFT_EYE)
        right_ear = self._compute_ear(lm, RIGHT_EYE)
        raw_ear = (left_ear + right_ear) / 2.0
        self._smoothed_ear = self.ema_alpha * raw_ear + (1 - self.ema_alpha) * self._smoothed_ear
        result.raw_ear = raw_ear
        result.ear = self._smoothed_ear

        eyes_closed = self._smoothed_ear < self.ear_threshold
        result.eyes_closed = eyes_closed

        # Track closed duration
        if eyes_closed:
            if self._eyes_closed_start is None:
                self._eyes_closed_start = time.time()
            result.closed_duration = time.time() - self._eyes_closed_start
        else:
            self._eyes_closed_start = None
            result.closed_duration = 0.0

        # Update PERCLOS window
        self._perclos_window.append(1 if eyes_closed else 0)
        if len(self._perclos_window) > 0:
            result.perclos = sum(self._perclos_window) / len(self._perclos_window)

        # ── MAR (Yawn) ───────────────────────────────────────────────
        raw_mar = self._compute_mar(lm)
        self._smoothed_mar = self.ema_alpha * raw_mar + (1 - self.ema_alpha) * self._smoothed_mar
        result.mar = self._smoothed_mar
        result.is_yawning = self._smoothed_mar > self.mar_threshold

        # ── Head Pose ────────────────────────────────────────────────
        yaw, pitch, roll = self._compute_head_pose(lm, w, h)
        result.yaw = yaw
        result.pitch = pitch
        result.roll = roll
        result.is_head_turned = abs(yaw) > self.head_yaw_threshold
        result.is_nodding = pitch > self.head_pitch_threshold

        # ── Counter logic (separated: eyes, head, yawn) ────────────
        if eyes_closed:
            self._closed_counter += 1
            self._open_counter = 0
        else:
            self._open_counter += 1
            self._closed_counter = max(0, self._closed_counter - 2)  # decay faster

        # Head turn — tracked independently, not mixed with eye closure
        if result.is_head_turned:
            self._head_turn_counter += 1
        else:
            self._head_turn_counter = max(0, self._head_turn_counter - 2)

        if result.is_nodding:
            self._nod_counter += 1
        else:
            self._nod_counter = max(0, self._nod_counter - 2)

        if result.is_yawning:
            self._yawn_counter += 1
        else:
            self._yawn_counter = max(0, self._yawn_counter - 1)

        # ── Determine if currently unsafe (any danger indicator) ──
        is_unsafe = (
            eyes_closed
            or result.is_yawning
            or result.is_head_turned
            or result.is_nodding
            or result.perclos > self.perclos_threshold
        )

        now = time.time()
        if is_unsafe:
            if self._unsafe_start_time is None:
                self._unsafe_start_time = now
            self._safe_start_time = None  # reset recovery timer
        else:
            if self._safe_start_time is None:
                self._safe_start_time = now
            # Don't clear unsafe timer immediately — allow brief safe
            # moments (< recovery_sec) without resetting
            if self._safe_start_time and (now - self._safe_start_time) >= self.recovery_sec:
                self._unsafe_start_time = None  # fully recovered

        # ── Build alert reason ───────────────────────────────────────
        reasons = []
        if eyes_closed:
            reasons.append(f"EYES CLOSED ({result.closed_duration:.1f}s)")
        if result.is_yawning:
            reasons.append("YAWNING")
        if result.is_head_turned:
            reasons.append(f"HEAD TURNED ({abs(yaw):.0f}°)")
        if result.is_nodding:
            reasons.append(f"NODDING ({pitch:.0f}°)")
        if result.perclos > self.perclos_threshold:
            reasons.append(f"HIGH PERCLOS ({result.perclos*100:.0f}%)")
        result.alert_reason = " | ".join(reasons) if reasons else "ALL CLEAR"

        # ── Confidence ───────────────────────────────────────────────
        danger_score = 0.0
        if eyes_closed:
            danger_score += 0.4
        if result.perclos > self.perclos_threshold:
            danger_score += 0.25
        if result.is_yawning:
            danger_score += 0.15
        if result.is_head_turned:
            danger_score += 0.1
        if result.is_nodding:
            danger_score += 0.1
            
        # ML Model refinement (if available)
        if self.model and result.face_detected:
            try:
                # Prepare features (matching train_model_99.py logic)
                feat = [
                    1, # simulated severity
                    result.face_bbox[0], result.face_bbox[1],
                    result.face_bbox[2], result.face_bbox[3],
                    result.face_bbox[2] * result.face_bbox[3], # Area
                    result.face_bbox[2] / (result.face_bbox[3] + 1e-6), # Aspect Ratio
                    0, # description hash placeholder
                    result.ear # Crucial real-world feature
                ]
                X = np.array([feat])
                if self.scaler:
                    X = self.scaler.transform(X)
                
                probs = self.model.predict_proba(X)[0]
                ml_confidence = probs.max()
                ml_class_idx = probs.argmax()
                ml_class = self.label_encoder.inverse_transform([ml_class_idx])[0]
                
                # Boost danger score if ML confirms drowsiness
                if ml_class in ['eyes_closed', 'drowsy_eyes', 'yawning']:
                    danger_score = max(danger_score, ml_confidence)
                    if ml_confidence > 0.8:
                        result.alert_reason += f" | AI: {ml_class.upper()}"
            except:
                pass

        result.confidence = min(1.0, danger_score)

        # ── State machine ────────────────────────────────────────────
        self._update_state(result)

        return result

    def _update_state(self, result: DetectionResult):
        """Time-based state machine with sustained detection.

        WARNING  triggers after `warning_delay_sec` (5s) of continuous unsafe.
        CRITICAL triggers after `critical_delay_sec` (7s) of continuous unsafe.
        Recovery requires `recovery_sec` (3s) of all-clear.
        """
        now = time.time()
        unsafe_duration = (
            (now - self._unsafe_start_time) if self._unsafe_start_time else 0.0
        )
        safe_duration = (
            (now - self._safe_start_time) if self._safe_start_time else 0.0
        )

        if self._state == DriverState.NORMAL:
            if unsafe_duration >= self.warning_delay_sec:
                self._state = DriverState.WARNING

        elif self._state == DriverState.WARNING:
            if unsafe_duration >= self.critical_delay_sec:
                self._state = DriverState.CRITICAL
            elif safe_duration >= self.recovery_sec:
                self._state = DriverState.NORMAL
                self._closed_counter = 0
                self._yawn_counter = 0
                self._head_turn_counter = 0
                self._nod_counter = 0

        elif self._state == DriverState.CRITICAL:
            # Require sustained recovery to exit critical
            if safe_duration >= self.recovery_sec * 2:
                self._state = DriverState.NORMAL
                self._closed_counter = 0
                self._yawn_counter = 0
                self._head_turn_counter = 0
                self._nod_counter = 0

        result.state = self._state

    def _compute_ear(self, landmarks: list, eye_indices: list) -> float:
        """Eye Aspect Ratio using 6 landmark points."""
        pts = [landmarks[i] for i in eye_indices]
        # Vertical distances
        A = dist.euclidean(pts[1], pts[5])
        B = dist.euclidean(pts[2], pts[4])
        # Horizontal distance
        C = dist.euclidean(pts[0], pts[3])
        if C < 1e-6:
            return 0.0
        return (A + B) / (2.0 * C)

    def _compute_mar(self, landmarks: list) -> float:
        """Mouth Aspect Ratio for yawn detection."""
        top = np.array(landmarks[MOUTH_INNER_TOP])
        bottom = np.array(landmarks[MOUTH_INNER_BOTTOM])
        left = np.array(landmarks[MOUTH_INNER_LEFT])
        right = np.array(landmarks[MOUTH_INNER_RIGHT])

        vertical = dist.euclidean(top, bottom)
        horizontal = dist.euclidean(left, right)
        if horizontal < 1e-6:
            return 0.0
        return vertical / horizontal

    def _compute_head_pose(self, landmarks: list, img_w: int, img_h: int) -> tuple:
        """Head pose estimation via solvePnP. Returns (yaw, pitch, roll) in degrees."""
        image_points = np.array([
            landmarks[idx] for idx in POSE_LANDMARKS
        ], dtype=np.float64)

        # Camera matrix (approximate)
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vec, translation_vec = cv2.solvePnP(
            self._model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rotation_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = angles[0]
        yaw = angles[1]
        roll = angles[2]

        return yaw, pitch, roll

    def reset(self):
        """Reset all state (e.g. when restarting detection)."""
        self._state = DriverState.NORMAL
        self._smoothed_ear = 0.30
        self._smoothed_mar = 0.10
        self._closed_counter = 0
        self._open_counter = 0
        self._yawn_counter = 0
        self._head_turn_counter = 0
        self._nod_counter = 0
        self._unsafe_start_time = None
        self._safe_start_time = None
        self._perclos_window.clear()
        self._eyes_closed_start = None

    def _load_ml_model(self):
        """Try to load the trained ensemble model."""
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        model_path = os.path.join(base_path, "driver_alertness_model.pkl")
        encoder_path = os.path.join(base_path, "label_encoder.pkl")
        scaler_path = os.path.join(base_path, "scaler.pkl")

        if os.path.exists(model_path) and os.path.exists(encoder_path):
            try:
                import joblib
                import pickle
                self.model = joblib.load(model_path)
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                print("✓ ML Neural Engine Loaded Successfully")
            except Exception as e:
                print(f"⚠ Neural Engine Load Warning: {e}")

    def release(self):
        """Release MediaPipe resources."""
        self.face_landmarker.close()
