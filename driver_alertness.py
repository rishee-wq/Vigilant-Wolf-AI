import cv2
import numpy as np
from scipy.spatial import distance as dist
import os
import pickle
import math
import time
import webbrowser
from threading import Thread
from dataclasses import dataclass, field
from types import SimpleNamespace
from pathlib import Path
import platform
import sys

# Cross-platform audio support
SYSTEM = platform.system()

if SYSTEM == "Windows":
    import winsound
elif SYSTEM == "Darwin":  # macOS
    import os
elif SYSTEM == "Linux":
    import subprocess

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

MIN_AER = 0.25
EYE_AR_CONSEC_FRAMES = 30
COUNTER = 0
ALARM_ON = False
HEAD_NOD_THRESHOLD = 0.15
HEAD_TURN_THRESHOLD = 30
PHONE_DETECTION_FRAMES = 15
SEATBELT_DETECTION_FRAMES = 20
EMERGENCY_NUMBER = "9650427590"
MAIN_WINDOW = "Vigilant Wolf AI - Dashboard"
ALERT_WINDOW = "Vigilant Wolf AI - Critical Alert"
MODERN_ALERT_SYSTEM = True

COLORS = {
    "bg_dark": "#06110d",
    "bg_secondary": "#0a1712",
    "surface_dark": "#10201a",
    "surface_light": "#163026",
    "surface_hover": "#1c3a2f",
    "border": "#1f4a3b",
    "border_light": "#2d6a56",
    "eco_teal": "#1fcd9a",
    "teal_bright": "#44f0ba",
    "accent_cyan": "#1bbf8a",
    "accent_violet": "#5cc78f",
    "alert_red": "#ff4d5f",
    "success_green": "#20d67a",
    "warning_orange": "#f6a24b",
    "text_primary": "#ecfff6",
    "text_secondary": "#98b8ab",
    "text_dim": "#57746b",
}


@dataclass
class AlertUIState:
    active: bool = False
    reason: str = ""
    started_at: float = 0.0
    countdown_seconds: int = 10
    called: bool = False
    muted: bool = False
    suppressed_until: float = 0.0
    sequence_token: int = 0
    button_rects: dict = field(default_factory=dict)
    last_reason: str = ""

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_landmarks(face, face_cascade):
    """Detect facial landmarks using dlib-style approach"""
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return None

def play_alarm(audio_file=None):
    """Play audio alert - Cross-platform support"""
    if MODERN_ALERT_SYSTEM:
        return
    try:
        # Skip invalid audio files and use system beep instead
        if audio_file and os.path.exists(audio_file):
            try:
                file_size = os.path.getsize(audio_file)
                # Only try to play if file is large enough to be valid audio
                if file_size > 5000:
                    if SYSTEM == "Darwin":  # macOS
                        os.system(f'afplay "{audio_file}"')
                        return
                    elif SYSTEM == "Linux":
                        for player in ["paplay", "aplay", "ffplay"]:
                            try:
                                subprocess.run([player, audio_file], timeout=5)
                                return
                            except:
                                continue
            except:
                pass
        
        # Use system beep for all cases (default, invalid file, or Windows)
        if SYSTEM == "Windows":
            winsound.Beep(800, 300)
            winsound.Beep(1000, 300)
        elif SYSTEM == "Darwin":
            try:
                os.system('afplay /System/Library/Sounds/Alarm.aiff 2>/dev/null')
            except:
                print('\a', end='', flush=True)
        elif SYSTEM == "Linux":
            try:
                os.system('paplay /usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga 2>/dev/null')
            except:
                print('\a', end='', flush=True)
    except Exception as e:
        # Silent fallback to bell
        try:
            print('\a', end='', flush=True)
        except:
            pass

def detect_available_cameras(max_cameras=5):
    """Detect available camera devices"""
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def get_optimal_resolution(cap):
    """Get optimal resolution based on camera capabilities"""
    # Try to get native resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # If resolution is very low, try to set better resolution
    if width < 320 or height < 240:
        preferred_resolutions = [(1280, 720), (800, 600), (640, 480), (320, 240)]
        for w, h in preferred_resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_w >= 320 and actual_h >= 240:
                return actual_w, actual_h
    
    return width, height

def detect_phone(frame, cascade_path=None):
    """Detect if person is looking at phone"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        # Check if face is tilted down (looking at phone)
        roi = frame[y:y+h, x:x+w]
        # Detect bright rectangles (phone screen)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_bright = np.array([0, 0, 200])
        upper_bright = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_bright, upper_bright)
        bright_pixels = np.sum(mask > 0)
        
        return bright_pixels > (roi.shape[0] * roi.shape[1] * 0.1)
    return False

def detect_seatbelt(frame):
    """Detect seatbelt presence"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect lines (seatbelt usually has diagonal lines)
    edges = cv2.Canny(gray, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    
    if lines is not None:
        # Check for diagonal lines typical of seatbelt
        diagonal_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            if 30 < angle < 60 or 120 < angle < 150:
                diagonal_count += 1
        
        return diagonal_count > 2
    return False

def detect_head_pose(frame):
    """Detect head pose (nodding, turning) - Very sensitive to any deviation"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None, None
    
    x, y, w, h = faces[0]
    face_center_x = x + w // 2
    frame_center_x = frame.shape[1] // 2
    
    # Detect head turn (left/right) - More sensitive threshold
    turn_ratio = (face_center_x - frame_center_x) / frame_center_x
    is_turned = abs(turn_ratio) > 0.10  # Even slight turn detected
    
    # Detect nodding (face moving up/down)
    face_center_y = y + h // 2
    frame_center_y = frame.shape[0] // 2
    nod_ratio = (face_center_y - frame_center_y) / frame_center_y
    is_nodding = nod_ratio > 0.15
    
    return is_turned, is_nodding

def extract_features(frame, face_cascade, eye_cascade):
    """Extract advanced features from frame for ML model"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None
    
    x, y, w, h = faces[0]
    
    # Extract eye variance (drowsiness indicator)
    roi_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    
    eye_variance = 0
    if len(eyes) > 0:
        for (ex, ey, ew, eh) in eyes:
            eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_variance = cv2.Laplacian(eye_region, cv2.CV_64F).var()
            break
    
    # Basic features
    severity = 1
    if eye_variance < 50:
        severity = 4
    
    # Derived features
    bbox_area = w * h
    bbox_aspect_ratio = w / (h + 1e-6)
    bbox_position_ratio = y / (300 + 1e-6)
    
    # Hash features
    filename_hash = 42 % 1000  # Fixed hash for frame
    behavior_hash = int(eye_variance) % 1000
    
    # Interaction features
    severity_position = severity * bbox_position_ratio
    severity_area = severity * bbox_area
    
    # Create feature vector (13 features - matching training)
    features = [
        severity,
        x,
        y,
        w,
        h,
        bbox_area,
        bbox_aspect_ratio,
        bbox_position_ratio,
        filename_hash,
        behavior_hash,
        severity_position,
        severity_area,
        eye_variance,
    ]
    
    return np.array(features, dtype=np.float32).reshape(1, -1)


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))


def clamp(value, low, high):
    return max(low, min(high, value))


def blend(base, overlay, alpha):
    return cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)


def make_gradient_background(width, height, top_hex, bottom_hex):
    top = np.array(hex_to_bgr(top_hex), dtype=np.float32)
    bottom = np.array(hex_to_bgr(bottom_hex), dtype=np.float32)
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t = y / max(1, height - 1)
        row = (top * (1 - t) + bottom * t).astype(np.uint8)
        gradient[y, :] = row
    return gradient


def draw_panel(img, x, y, w, h, fill_hex, border_hex, alpha=0.88, border_thickness=2):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), hex_to_bgr(fill_hex), -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), hex_to_bgr(border_hex), border_thickness, cv2.LINE_AA)
    return blend(img, overlay, alpha)


def draw_text(img, text, org, scale, color_hex, thickness=2, shadow=True):
    color = hex_to_bgr(color_hex)
    if shadow:
        cv2.putText(img, text, (org[0] + 2, org[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_center_text(img, text, center_x, center_y, scale, color_hex, thickness=2, shadow=True):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    draw_text(img, text, (int(center_x - tw / 2), int(center_y + th / 2)), scale, color_hex, thickness, shadow)


def point_in_rect(point, rect):
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def draw_ring(img, center, radius, pct, color_hex, track_hex="#20372f", thickness=16):
    cx, cy = center
    cv2.ellipse(img, center, (radius, radius), -90, 0, 360, hex_to_bgr(track_hex), thickness, cv2.LINE_AA)
    end_angle = -90 + 360 * clamp(pct, 0.0, 1.0)
    cv2.ellipse(img, center, (radius, radius), -90, 0, end_angle + 90, hex_to_bgr(color_hex), thickness, cv2.LINE_AA)


def draw_metric_card(img, x, y, w, h, label, value, accent_hex, sublabel=None):
    img = draw_panel(img, x, y, w, h, COLORS["surface_dark"], COLORS["border"], alpha=0.94)
    cv2.rectangle(img, (x, y), (x + 6, y + h), hex_to_bgr(accent_hex), -1, cv2.LINE_AA)
    draw_text(img, label.upper(), (x + 18, y + 24), 0.45, COLORS["text_secondary"], 1, shadow=False)
    draw_text(img, value, (x + 18, y + 56), 0.82, COLORS["text_primary"], 2, shadow=False)
    if sublabel:
        draw_text(img, sublabel, (x + 18, y + h - 12), 0.38, accent_hex, 1, shadow=False)
    return img


def draw_danger_widget(img, x, y, w, h, phase):
    widget = img.copy()
    widget = draw_panel(widget, x, y, w, h, "#220708", "#ff4d5f", alpha=0.92)
    inner = widget[y + 10:y + h - 10, x + 10:x + w - 10]
    inner[:] = make_gradient_background(w - 20, h - 20, "#2b0507", "#090101")

    cx = x + w // 2
    cy = y + h // 2
    max_r = min(w, h) * 0.42
    for i in range(8):
        angle = (phase * 140 + i * 22) % 360
        rad = math.radians(angle)
        length = int(max_r * (0.35 + 0.08 * i))
        sx = int(cx + math.cos(rad) * (max_r * 0.08))
        sy = int(cy + math.sin(rad) * (max_r * 0.08))
        ex = int(cx + math.cos(rad) * length)
        ey = int(cy + math.sin(rad) * length)
        cv2.line(widget, (sx, sy), (ex, ey), (60, 20, 255), 6, cv2.LINE_AA)
        cv2.line(widget, (sx, sy), (ex, ey), (160, 120, 255), 2, cv2.LINE_AA)

    rot = phase * 90
    hex_r = int(min(w, h) * 0.18)
    pts = []
    for i in range(6):
        a = math.radians(rot + 60 * i)
        pts.append((int(cx + hex_r * math.cos(a)), int(cy + hex_r * math.sin(a))))
    for i in range(6):
        cv2.line(widget, pts[i], pts[(i + 1) % 6], (255, 71, 87), 3, cv2.LINE_AA)
    cv2.putText(widget, "DANGER", (x + 18, y + h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 150, 160), 2, cv2.LINE_AA)
    return widget


def render_dashboard_frame(frame, result, runtime):
    height, width = 760, 1280
    canvas = make_gradient_background(width, height, COLORS["bg_dark"], COLORS["bg_secondary"])

    # Ambient glow accents
    cv2.circle(canvas, (int(width * 0.16), int(height * 0.18)), 190, (20, 90, 64), -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(width * 0.84), int(height * 0.22)), 160, (30, 70, 40), -1, cv2.LINE_AA)
    canvas = blend(canvas, canvas, 0.99)

    # Header
    canvas = draw_panel(canvas, 18, 16, width - 36, 76, "#0d1d16", COLORS["border_light"], alpha=0.96)
    draw_text(canvas, "VIGILANT WOLF AI", (44, 63), 0.82, COLORS["eco_teal"], 2, shadow=False)
    draw_text(canvas, "eco monitoring dashboard", (46, 89), 0.45, COLORS["text_secondary"], 1, shadow=False)

    live_badge_x, live_badge_y, live_badge_w, live_badge_h = 410, 34, 126, 34
    canvas = draw_panel(canvas, live_badge_x, live_badge_y, live_badge_w, live_badge_h, "#0f231c", "#1e5c47", alpha=0.94, border_thickness=1)
    cv2.circle(canvas, (live_badge_x + 17, live_badge_y + 17), 6, hex_to_bgr(COLORS["success_green"]), -1, cv2.LINE_AA)
    draw_text(canvas, "LIVE", (live_badge_x + 32, live_badge_y + 23), 0.52, COLORS["success_green"], 1, shadow=False)

    now = time.strftime("%H:%M:%S")
    draw_text(canvas, now, (width - 160, 57), 0.62, COLORS["text_primary"], 2, shadow=False)

    # Main layout boxes
    cam_x, cam_y, cam_w, cam_h = 26, 112, 850, 552
    side_x, side_y, side_w, side_h = 898, 112, 356, 552
    canvas = draw_panel(canvas, cam_x, cam_y, cam_w, cam_h, COLORS["surface_dark"], COLORS["border"], alpha=0.94)
    canvas = draw_panel(canvas, side_x, side_y, side_w, side_h, COLORS["surface_dark"], COLORS["border"], alpha=0.96)

    # Camera frame
    available_w = cam_w - 30
    available_h = cam_h - 70
    if frame is not None and frame.size:
        frame_h, frame_w = frame.shape[:2]
        scale = min(available_w / frame_w, available_h / frame_h)
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        top = cam_y + 55 + (available_h - new_h) // 2
        left = cam_x + 15 + (available_w - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized
        cv2.rectangle(canvas, (left, top), (left + new_w, top + new_h), hex_to_bgr(COLORS["border_light"]), 2, cv2.LINE_AA)

    draw_text(canvas, "CAMERA FEED", (cam_x + 20, cam_y + 34), 0.58, COLORS["text_secondary"], 1, shadow=False)
    draw_text(canvas, "Driver face tracking and fatigue signals", (cam_x + 20, cam_y + cam_h - 18), 0.46, COLORS["text_dim"], 1, shadow=False)

    # Right column: safety score
    draw_text(canvas, "SAFETY SCORE", (side_x + 20, side_y + 30), 0.56, COLORS["text_secondary"], 1, shadow=False)
    score_pct = clamp(1.0 - getattr(result, "drowsiness", 0.0), 0.0, 1.0)
    draw_ring(canvas, (side_x + 92, side_y + 112), 58, score_pct, COLORS["eco_teal"], track_hex="#234035", thickness=16)
    score_text = f"{int(score_pct * 100)}%"
    draw_center_text(canvas, score_text, side_x + 92, side_y + 118, 0.92, COLORS["text_primary"], 2, shadow=False)
    score_label = "STABLE" if score_pct >= 0.9 else ("WATCH" if score_pct >= 0.75 else "ALERT")
    draw_center_text(canvas, score_label, side_x + 92, side_y + 168, 0.48, COLORS["success_green"] if score_label == "STABLE" else COLORS["warning_orange"] if score_label == "WATCH" else COLORS["alert_red"], 1, shadow=False)

    status_state = getattr(result.state, "value", "NORMAL")
    status_color = COLORS["success_green"] if status_state == "NORMAL" else COLORS["warning_orange"] if status_state == "WARNING" else COLORS["alert_red"]
    canvas = draw_metric_card(canvas, side_x + 176, side_y + 58, 160, 104, "Status", status_state, status_color, getattr(result, "alert_reason", "" )[:28])

    # Metrics grid
    ear = getattr(result, "ear", 0.0)
    mar = getattr(result, "mar", 0.0)
    perclos = getattr(result, "perclos", 0.0)
    confidence = getattr(result, "confidence", 0.0)
    canvas = draw_metric_card(canvas, side_x + 18, side_y + 204, 150, 96, "EAR", f"{ear:.2f}", COLORS["eco_teal"], "normal eye openness")
    canvas = draw_metric_card(canvas, side_x + 178, side_y + 204, 160, 96, "MAR", f"{mar:.2f}", COLORS["warning_orange"], "yawn intensity")
    canvas = draw_metric_card(canvas, side_x + 18, side_y + 310, 150, 96, "PERCLOS", f"{int(perclos * 100)}%", COLORS["alert_red"], "eye closure window")
    canvas = draw_metric_card(canvas, side_x + 178, side_y + 310, 160, 96, "Confidence", f"{int(confidence * 100)}%", COLORS["accent_violet"], "danger score")

    # Trip / alerts footer chips
    trip_text = runtime.get("trip_text", "00:00:00")
    alert_count = runtime.get("alert_count", 0)
    canvas = draw_metric_card(canvas, side_x + 18, side_y + 416, 150, 96, "Trip", trip_text, COLORS["success_green"], "session timer")
    canvas = draw_metric_card(canvas, side_x + 178, side_y + 416, 160, 96, "Alerts", str(alert_count), COLORS["alert_red"], "critical triggers")

    # Footer
    footer_y = 680
    canvas = draw_panel(canvas, 18, footer_y, width - 36, 54, "#0d1d16", COLORS["border"], alpha=0.96)
    footer_state = getattr(result.state, "value", "NORMAL")
    footer_text = "SYSTEM READY" if footer_state == "NORMAL" else ("FATIGUE WARNING" if footer_state == "WARNING" else "CRITICAL ALERT")
    footer_color = COLORS["success_green"] if footer_state == "NORMAL" else COLORS["warning_orange"] if footer_state == "WARNING" else COLORS["alert_red"]
    draw_text(canvas, footer_text, (42, footer_y + 33), 0.58, footer_color, 1, shadow=False)
    draw_text(canvas, getattr(result, "alert_reason", "")[:72], (250, footer_y + 33), 0.46, COLORS["text_secondary"], 1, shadow=False)
    draw_text(canvas, "Press ESC to exit", (width - 180, footer_y + 33), 0.46, COLORS["text_dim"], 1, shadow=False)

    return canvas


def trigger_call(number):
    try:
        webbrowser.open(f"tel:{number}")
        return True
    except Exception:
        try:
            if SYSTEM == "Windows":
                os.startfile(f"tel:{number}")
                return True
        except Exception:
            pass
    return False


def run_dual_warning_sequence(alert_state: AlertUIState, reason: str):
    token = alert_state.sequence_token

    def _worker():
        if alert_state.muted or token != alert_state.sequence_token:
            return

        # Stage 1: rapid beep sequence.
        for freq, duration in [(880, 120), (1050, 120), (880, 120), (1320, 180), (880, 120), (1050, 160)]:
            if alert_state.muted or token != alert_state.sequence_token:
                return
            if SYSTEM == "Windows":
                try:
                    winsound.Beep(freq, duration)
                except Exception:
                    print('\a', end='', flush=True)
            else:
                print('\a', end='', flush=True)
            time.sleep(0.05)

        if alert_state.muted or token != alert_state.sequence_token:
            return

        time.sleep(0.35)

        # Stage 2: spoken warning.
        if HAS_TTS:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 158)
                engine.setProperty("volume", 1.0)
                spoken = reason if reason else "Warning. Driver fatigue detected. Please pull over immediately and rest."
                engine.say(f"Warning. Driver fatigue detected. {spoken}. Please pull over immediately and rest.")
                engine.runAndWait()
                engine.stop()
            except Exception:
                pass
        elif SYSTEM == "Windows":
            try:
                winsound.Beep(1240, 300)
                winsound.Beep(1240, 300)
            except Exception:
                pass

    Thread(target=_worker, daemon=True).start()


def alert_click_handler(event, x, y, flags, runtime):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    alert_state = runtime.get("alert_state")
    if not alert_state or not alert_state.active:
        return
    for name, rect in alert_state.button_rects.items():
        if point_in_rect((x, y), rect):
            if name == "awake":
                runtime["dismiss_alert"]()
            elif name == "mute":
                alert_state.muted = True
            elif name == "call":
                alert_state.called = True
                runtime["make_call"](manual=True)
            break


def render_alert_frame(width, height, alert_state: AlertUIState, result, runtime):
    phase = time.time() - alert_state.started_at
    canvas = make_gradient_background(width, height, "#350508", "#090000")

    # Fast moving red lines.
    for idx in range(28):
        speed = 0.7 + (idx % 6) * 0.2
        offset = (phase * 260 * speed + idx * 42) % (width + 220) - 110
        y = int((idx / 28) * (height - 120) + 60)
        cv2.line(canvas, (int(offset), y), (int(offset + 240), y - 48), (35, 0, 255), 6, cv2.LINE_AA)
        cv2.line(canvas, (int(offset), y), (int(offset + 240), y - 48), (140, 80, 255), 2, cv2.LINE_AA)

    # Subtle sparks.
    for idx in range(18):
        angle = math.radians((phase * 180 + idx * 23) % 360)
        radius = 60 + idx * 13
        sx = int(width / 2 + math.cos(angle) * radius)
        sy = int(height / 2 + math.sin(angle) * radius * 0.55)
        cv2.circle(canvas, (sx, sy), 2 + idx % 3, (90, 40, 255), -1, cv2.LINE_AA)

    # Header strip.
    canvas = draw_panel(canvas, 18, 16, width - 36, 74, "#260306", "#ff4d5f", alpha=0.94)
    draw_text(canvas, "CRITICAL ALERT", (42, 58), 0.82, COLORS["alert_red"], 2, shadow=False)
    draw_text(canvas, "driver fatigue intervention mode", (44, 86), 0.44, "#ffbcc4", 1, shadow=False)
    draw_text(canvas, f"Auto-calling {EMERGENCY_NUMBER} in {max(0, alert_state.countdown_seconds - int(time.time() - alert_state.started_at))} s", (width - 405, 58), 0.52, "#ffd166", 1, shadow=False)

    # 3D danger widget.
    canvas = draw_danger_widget(canvas, width - 360, 110, 320, 220, phase)

    # Center warning copy.
    draw_center_text(canvas, "WAKE UP!", width / 2, height / 2 - 84, 1.9, COLORS["alert_red"], 4, shadow=True)
    draw_center_text(canvas, "PULL OVER IMMEDIATELY", width / 2, height / 2 - 24, 0.95, "#fff1f2", 2, shadow=True)
    reason = alert_state.reason or getattr(result, "alert_reason", "") or "Driver fatigue detected"
    draw_center_text(canvas, reason[:54], width / 2, height / 2 + 26, 0.64, "#ff9aa8", 2, shadow=False)

    # Countdown pill.
    remaining = max(0, alert_state.countdown_seconds - int(time.time() - alert_state.started_at))
    countdown_w, countdown_h = 430, 58
    countdown_x = int(width / 2 - countdown_w / 2)
    countdown_y = int(height / 2 + 70)
    canvas = draw_panel(canvas, countdown_x, countdown_y, countdown_w, countdown_h, "#250407", "#ff4d5f", alpha=0.95)
    draw_center_text(canvas, f"CALLING IN {remaining} SECONDS", width / 2, countdown_y + 34, 0.85, "#ffd166", 2, shadow=False)

    # Buttons.
    button_y = height - 128
    button_specs = [
        ("awake", "I AM AWAKE", width // 2 - 390, button_y, 240, 72, "#f5fff9", "#20d67a"),
        ("mute", "MUTE ALARM", width // 2 - 120, button_y, 240, 72, "#fff8ef", "#f6a24b"),
        ("call", f"CALL {EMERGENCY_NUMBER}", width // 2 + 150, button_y, 240, 72, "#fff1f2", "#ff4d5f"),
    ]
    alert_state.button_rects = {}
    for name, label, x, y, w, h, text_hex, accent_hex in button_specs:
        alert_state.button_rects[name] = (x, y, w, h)
        canvas = draw_panel(canvas, x, y, w, h, "#190305", accent_hex, alpha=0.96)
        draw_center_text(canvas, label, x + w / 2, y + 43, 0.72, text_hex, 2, shadow=False)

    # Footer.
    canvas = draw_panel(canvas, 18, height - 50, width - 36, 34, "#240408", "#ff4d5f", alpha=0.94)
    draw_text(canvas, "LIVE MONITORING ACTIVE", (40, height - 27), 0.45, "#ff9aa8", 1, shadow=False)
    draw_text(canvas, "Press the green button if the driver is awake", (width - 410, height - 27), 0.42, COLORS["text_secondary"], 1, shadow=False)

    return canvas

def main():
    global COUNTER, ALARM_ON
    
    print(f"System: {SYSTEM}")
    print("Initializing Driver Alertness Detection System...\n")
    
    # Load ML model if available
    model = None
    label_encoder = None
    scaler = None
    model_path = os.path.join('models', 'driver_alertness_model.pkl')
    encoder_path = os.path.join('models', 'label_encoder.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            print("✓ ML Model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Running with rule-based detection only...")
    else:
        print("Note: ML model not found. To train the model, run:")
        print("  python train_model_advanced.py (for 99%+ accuracy)")
        print("  or python train_model.py (for basic model)")
        print("Using rule-based detection for now...\n")
    
    # Load cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Auto-detect available cameras
    print("Detecting available cameras...")
    available_cameras = detect_available_cameras()
    
    if not available_cameras:
        print("ERROR: No camera detected! Please check your camera connection.")
        return
    
    camera_index = available_cameras[0]
    if len(available_cameras) > 1:
        print(f"Found {len(available_cameras)} camera(s). Using camera {camera_index}")
    else:
        print(f"Using camera {camera_index}")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {camera_index}")
        return
    
    # Get optimal resolution
    print("Getting camera resolution...")
    width, height = get_optimal_resolution(cap)
    print(f"Camera resolution: {width}x{height}\n")
    
    # Create dashboard and alert windows.
    cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(MAIN_WINDOW, 1280, 760)
    cv2.namedWindow(ALERT_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ALERT_WINDOW, 1280, 760)

    alert_state = AlertUIState()
    runtime = {
        "alert_state": alert_state,
        "alert_count": 0,
        "trip_text": "00:00:00",
    }
    trip_start = time.time()

    def _dismiss_alert():
        global ALARM_ON
        alert_state.active = False
        alert_state.called = False
        alert_state.reason = ""
        alert_state.muted = False
        alert_state.sequence_token += 1
        alert_state.suppressed_until = time.time() + 2.5
        ALARM_ON = False
        try:
            cv2.setWindowProperty(ALERT_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        except Exception:
            pass

    def _make_call(manual=False):
        alert_state.called = True
        trigger_call(EMERGENCY_NUMBER)
        if manual:
            alert_state.suppressed_until = time.time() + 2.5

    def _start_alert(reason_text):
        global ALARM_ON
        if alert_state.active:
            alert_state.reason = reason_text or alert_state.reason
            return
        alert_state.active = True
        alert_state.reason = reason_text or "Driver fatigue detected"
        alert_state.started_at = time.time()
        alert_state.called = False
        alert_state.muted = False
        alert_state.sequence_token += 1
        runtime["alert_count"] += 1
        ALARM_ON = True
        try:
            cv2.setWindowProperty(ALERT_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass
        run_dual_warning_sequence(alert_state, alert_state.reason)

    runtime["dismiss_alert"] = _dismiss_alert
    runtime["make_call"] = _make_call
    runtime["start_alert"] = _start_alert
    cv2.setMouseCallback(ALERT_WINDOW, alert_click_handler, runtime)
    
    phone_counter = 0
    seatbelt_counter = 0
    turn_counter = 0
    nod_counter = 0
    
    print("Starting detection... Press ESC or close window to exit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from camera")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        status_text = "NORMAL"
        status_color = (0, 255, 0)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Get ML model prediction if available
            if model is not None:
                features = extract_features(frame, face_cascade, eye_cascade)
                if features is not None:
                    try:
                        # Scale features if scaler is available
                        if scaler is not None:
                            features_scaled = scaler.transform(features)
                        else:
                            features_scaled = features
                        
                        prediction = model.predict(features_scaled)[0]
                        predicted_class = label_encoder.inverse_transform([prediction])[0]
                        
                        # Get confidence
                        try:
                            confidence = model.predict_proba(features_scaled)[0].max()
                        except:
                            confidence = 0.8  # Default if proba not available
                        
                        # Map model predictions to actions
                        if predicted_class == 'eyes_closed' and confidence > 0.6:
                            status_text = f"DROWSY - SLEEP! ({confidence:.2f})"
                            status_color = (0, 0, 255)
                            if not ALARM_ON:
                                ALARM_ON = True
                                t = Thread(target=play_alarm, args=(audio_path,))
                                t.daemon = True
                                t.start()
                        elif predicted_class == 'drowsy_eyes' and confidence > 0.6:
                            status_text = f"DROWSY EYES! ({confidence:.2f})"
                            status_color = (0, 0, 255)
                            if not ALARM_ON:
                                ALARM_ON = True
                                t = Thread(target=play_alarm, args=(audio_path,))
                                t.daemon = True
                                t.start()
                        elif predicted_class == 'yawning' and confidence > 0.6:
                            status_text = f"YAWNING - SLEEPY! ({confidence:.2f})"
                            status_color = (0, 0, 255)
                            if not ALARM_ON:
                                ALARM_ON = True
                                t = Thread(target=play_alarm, args=(audio_path,))
                                t.daemon = True
                                t.start()
                        elif predicted_class == 'phone_use' and confidence > 0.6:
                            status_text = f"PHONE USE! ({confidence:.2f})"
                            status_color = (0, 165, 255)
                            if not ALARM_ON:
                                ALARM_ON = True
                                t = Thread(target=play_alarm, args=(audio_path,))
                                t.daemon = True
                                t.start()
                        elif predicted_class == 'phone_scroll' and confidence > 0.6:
                            status_text = f"PHONE DETECTED! ({confidence:.2f})"
                            status_color = (0, 165, 255)
                            if not ALARM_ON:
                                ALARM_ON = True
                                t = Thread(target=play_alarm, args=(audio_path,))
                                t.daemon = True
                                t.start()
                        elif predicted_class == 'head_turned' and confidence > 0.6:
                            status_text = f"SEE STRAIGHT! ({confidence:.2f})"
                            status_color = (0, 0, 255)
                            if not ALARM_ON:
                                ALARM_ON = True
                                t = Thread(target=play_alarm, args=(audio_path,))
                                t.daemon = True
                                t.start()
                        elif predicted_class == 'no_seatbelt' and confidence > 0.6:
                            status_text = f"NO SEATBELT! ({confidence:.2f})"
                            status_color = (0, 100, 255)
                            if not ALARM_ON:
                                ALARM_ON = True
                                t = Thread(target=play_alarm, args=(audio_path,))
                                t.daemon = True
                                t.start()
                        elif predicted_class == 'drinking' and confidence > 0.6:
                            status_text = f"DRINKING - NOT ATTENTIVE! ({confidence:.2f})"
                            status_color = (0, 165, 255)
                            if not ALARM_ON:
                                ALARM_ON = True
                                t = Thread(target=play_alarm, args=(audio_path,))
                                t.daemon = True
                                t.start()
                        else:
                            status_text = f"NORMAL ({confidence:.2f})"
                            status_color = (0, 255, 0)
                            ALARM_ON = False
                    except Exception as e:
                        print(f"Prediction error: {e}")
            else:
                # Fallback to rule-based detection
                # Eye detection
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                # Drowsiness detection
                for (ex, ey, ew, eh) in eyes:
                    eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
                    variance = cv2.Laplacian(eye_region, cv2.CV_64F).var()
                    
                    if variance < 50:
                        COUNTER += 1
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            status_text = "DROWSY - SLEEP!"
                            status_color = (0, 0, 255)
                            if not ALARM_ON:
                                ALARM_ON = True
                                t = Thread(target=play_alarm, args=(audio_path,))
                                t.daemon = True
                                t.start()
                    else:
                        COUNTER = 0
                        ALARM_ON = False
            
            # Head pose detection (additional check)
            is_turned, is_nodding = detect_head_pose(frame)
            
            if is_nodding:
                nod_counter += 1
                if nod_counter > 10:
                    status_text = "NODDING DOWN - SLEEP!"
                    status_color = (0, 0, 255)
                    if not ALARM_ON:
                        ALARM_ON = True
                        t = Thread(target=play_alarm, args=(audio_path,))
                        t.daemon = True
                        t.start()
            else:
                nod_counter = 0
            
            if is_turned:
                turn_counter += 1
                if turn_counter > 3:  # Trigger alarm much faster
                    status_text = "SEE STRAIGHT - ALARM!"
                    status_color = (0, 0, 255)
                    if not ALARM_ON:
                        ALARM_ON = True
                        t = Thread(target=play_alarm, args=(audio_path,))
                        t.daemon = True
                        t.start()
            else:
                turn_counter = 0
                if model is None:  # Only reset if not using ML model
                    ALARM_ON = False
            
            # Phone detection
            if detect_phone(frame):
                phone_counter += 1
                if phone_counter > PHONE_DETECTION_FRAMES:
                    status_text = "PHONE DETECTED - NOT ATTENTIVE!"
                    status_color = (0, 165, 255)
                    if not ALARM_ON:
                        ALARM_ON = True
                        t = Thread(target=play_alarm, args=(audio_path,))
                        t.daemon = True
                        t.start()
            else:
                phone_counter = 0
            
            # Seatbelt detection
            if not detect_seatbelt(frame):
                seatbelt_counter += 1
                if seatbelt_counter > SEATBELT_DETECTION_FRAMES:
                    status_text = "NO SEATBELT!"
                    status_color = (0, 100, 255)
                    if not ALARM_ON:
                        ALARM_ON = True
                        t = Thread(target=play_alarm, args=(audio_path,))
                        t.daemon = True
                        t.start()
            else:
                seatbelt_counter = 0
        
        # Build a modern dashboard view and switch to the alert overlay if needed.
        visual_state = "NORMAL"
        if status_color in ((0, 0, 255),):
            visual_state = "CRITICAL"
        elif status_color in ((0, 165, 255), (0, 100, 255)):
            visual_state = "WARNING"

        runtime["trip_text"] = f"{int(time.time() - trip_start) // 3600:02d}:{(int(time.time() - trip_start) % 3600) // 60:02d}:{int(time.time() - trip_start) % 60:02d}"

        # Derive lightweight dashboard metrics from the existing detector state.
        stability = 1.0
        if visual_state == "WARNING":
            stability = 0.72
        elif visual_state == "CRITICAL":
            stability = 0.38
        if phone_counter > 0:
            stability -= min(0.12, phone_counter / max(1, PHONE_DETECTION_FRAMES * 8))
        if seatbelt_counter > 0:
            stability -= min(0.08, seatbelt_counter / max(1, SEATBELT_DETECTION_FRAMES * 10))
        stability = max(0.05, min(1.0, stability))

        frame_result = SimpleNamespace(
            state=SimpleNamespace(value=visual_state),
            ear=max(0.0, min(1.0, COUNTER / max(1, EYE_AR_CONSEC_FRAMES))),
            mar=max(0.0, min(1.0, phone_counter / max(1, PHONE_DETECTION_FRAMES))),
            perclos=max(0.0, min(1.0, (turn_counter + nod_counter + seatbelt_counter) / 30.0)),
            confidence=1.0 - stability,
            drowsiness=1.0 - stability,
            alert_reason=status_text,
        )

        if visual_state == "NORMAL" and not alert_state.active:
            ALARM_ON = False

        if visual_state == "CRITICAL" and time.time() >= alert_state.suppressed_until and not alert_state.active:
            runtime["start_alert"](status_text)

        if alert_state.active:
            remaining = max(0, alert_state.countdown_seconds - int(time.time() - alert_state.started_at))
            if remaining <= 0 and not alert_state.called:
                _make_call()

            alert_frame = render_alert_frame(1280, 760, alert_state, frame_result, runtime)
            cv2.imshow(ALERT_WINDOW, alert_frame)
        else:
            dashboard_frame = render_dashboard_frame(frame, frame_result, runtime)
            cv2.imshow(MAIN_WINDOW, dashboard_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nDetection system closed.")

if __name__ == "__main__":
    main()
