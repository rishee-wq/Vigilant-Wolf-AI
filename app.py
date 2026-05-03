"""
Vigilant Wolf AI — Intelligent Driver Fatigue Detection  (v3.0)

Three-window system:
  0. Startup: Geometric wolf logo animation with automated loading bar.
  1. Dashboard: EV-themed dark teal/charcoal monitoring view with live camera,
     EAR/MAR gauges, PERCLOS, trip timer, alert history, safety score.
  2. Alert Overlay: full-screen red glowing alert with rapidly-moving
     animated danger lines, dual auditory cues (beep + TTS), and a
     10-second auto-call countdown to 9650427590.

Usage:
    python app.py
"""

import sys
import os
import time
import math
import random
import platform
import threading
import webbrowser
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QSizePolicy,
    QGraphicsDropShadowEffect, QDesktopWidget, QStackedLayout,
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation,
    QEasingCurve, QSize, QRect, QRectF, pyqtProperty, QUrl, QPointF,
)
from PyQt5.QtGui import (
    QImage, QPixmap, QColor, QPainter, QFont, QFontDatabase,
    QPen, QBrush, QLinearGradient, QRadialGradient, QPalette,
    QIcon, QConicalGradient,
)

import cv2
import numpy as np

from detection_engine import DetectionEngine, DetectionResult, DriverState

try:
    from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
    HAS_MULTIMEDIA = True
except ImportError:
    HAS_MULTIMEDIA = False

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

SYSTEM = platform.system()

# ─── Premium Dark Colour Palette ─────────────────────────────────────

COLORS = {
    "bg_dark":        "#050705",
    "bg_secondary":   "#010201",
    "surface_dark":   "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(20, 35, 20, 0.8), stop:1 rgba(5, 10, 5, 0.9))",
    "surface_light":  "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(40, 70, 40, 0.7), stop:1 rgba(15, 25, 15, 0.8))",
    "surface_hover":  "rgba(50, 100, 50, 0.9)",
    "border":         "rgba(34, 197, 94, 0.15)",
    "border_light":   "rgba(34, 197, 94, 0.3)",
    "eco_green":      "#22c55e",
    "green_bright":   "#4ade80",
    "accent_emerald": "#10b981",
    "accent_lime":    "#84cc16",
    "alert_red":      "#ef4444",
    "success_green":  "#22c55e",
    "warning_orange": "#f59e0b",
    "text_primary":   "#f0fdf4",
    "text_secondary": "#86efac",
    "text_dim":       "#166534",
}

FONT_FAMILY = "Segoe UI"
FONT_MONO = "Consolas"
EMERGENCY_NUMBER = "9650427590"

def add_shadow(widget, blur_radius=20, x_offset=0, y_offset=8, color="#000000", alpha=120):
    """Helper to add a soft drop shadow to a QWidget."""
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(blur_radius)
    shadow.setXOffset(x_offset)
    shadow.setYOffset(y_offset)
    shadow.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(shadow)



# ═══════════════════════════════════════════════════════════════════════
#  Geometric Fox Startup Animation
# ═══════════════════════════════════════════════════════════════════════

class StartupWolfWindow(QWidget):
    """
    Frameless startup splash with a geometric/low-poly wolf head drawn
    progressively from lines, followed by an automated loading bar.
    """
    finished = pyqtSignal()   # emitted when loading completes

    # ── Fox vertex geometry (normalised –1…+1, will be scaled) ──────
    # Each entry is ((x1,y1), (x2,y2))  — one line segment of the fox.
    WOLF_LINES = [
        # ── Ear ────────────────────────────────────────────────────────
        ((-0.50, -0.20), (-0.40, -0.70)), # back of ear
        ((-0.40, -0.70), (-0.20, -0.45)), # front of ear
        ((-0.50, -0.20), (-0.20, -0.45)), # ear base
        
        # ── Head Crown / Forehead ──────────────────────────────────────
        ((-0.20, -0.45), (-0.05, -0.55)), # crown to forehead
        ((-0.05, -0.55), (0.25, -0.70)),  # forehead to snout bridge
        
        # ── Snout & Nose ───────────────────────────────────────────────
        ((0.25, -0.70), (0.60, -0.85)),   # bridge to nose tip
        ((0.60, -0.85), (0.65, -0.80)),   # nose tip to upper lip
        ((0.65, -0.80), (0.20, -0.45)),   # upper lip to mouth corner
        
        # ── Lower Jaw (Howling Open Mouth) ─────────────────────────────
        ((0.20, -0.45), (0.50, -0.60)),   # mouth corner to lower jaw tip
        ((0.50, -0.60), (0.10, -0.25)),   # lower jaw tip to base/throat
        
        # ── Neck & Chest (Front) ───────────────────────────────────────
        ((0.10, -0.25), (0.35, 0.15)),    # throat to mid neck
        ((0.35, 0.15), (0.30, 0.65)),     # mid neck to front chest
        
        # ── Neck & Back (Rear) ─────────────────────────────────────────
        ((-0.50, -0.20), (-0.65, 0.10)),  # ear base to back of neck
        ((-0.65, 0.10), (-0.75, 0.60)),   # back of neck to lower back
        
        # ── Internal Facets (Low Poly Effect) ──────────────────────────
        # Eye area
        ((-0.05, -0.55), (0.05, -0.40)),  # forehead to eye
        ((0.05, -0.40), (0.20, -0.45)),   # eye to mouth corner
        ((0.05, -0.40), (-0.20, -0.45)),  # eye to ear base front
        ((-0.50, -0.20), (0.05, -0.40)),  # ear base back to eye
        
        # Cheek / Jaw muscle
        ((0.05, -0.40), (0.10, -0.25)),   # eye to lower jaw base
        ((-0.20, -0.45), (0.10, -0.25)),  # ear front to lower jaw base
        ((-0.50, -0.20), (0.10, -0.25)),  # ear back to jaw base
        ((-0.65, 0.10), (0.10, -0.25)),   # back neck to jaw base
        
        # Neck muscle definitions
        ((-0.65, 0.10), (0.35, 0.15)),    # upper back to mid front neck
        ((-0.75, 0.60), (0.35, 0.15)),    # back to mid front neck
        ((-0.75, 0.60), (0.30, 0.65)),    # lower back straight across to lower chest (base)
        
        # Snout detailing
        ((0.25, -0.70), (0.20, -0.45)),   # snout bridge down to mouth corner
        ((0.05, -0.40), (0.25, -0.70)),   # eye to snout bridge
    ]

    LOADING_MESSAGES = [
        "Initializing Neural Engine…",
        "Loading Face Landmark Model…",
        "Calibrating EAR / MAR Thresholds…",
        "Warming Up Detection Pipeline…",
        "Preparing Dashboard Interface…",
        "System Ready — Launching…",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(720, 620)
        
        # Load and scale the image
        img_path = os.path.join(os.path.dirname(__file__), "assets", "wolf_logo.png")
        if os.path.exists(img_path):
            self.logo_pixmap = QPixmap(img_path).scaled(
                340, 340, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        else:
            self.logo_pixmap = None
            
        # Pre-calculate assembly chunks
        self.chunks = []
        if self.logo_pixmap:
            img_w = self.logo_pixmap.width()
            img_h = self.logo_pixmap.height()
            cols = 16
            rows = 16
            cw = img_w / cols
            ch = img_h / rows
            for r in range(rows):
                for c in range(cols):
                    tx = c * cw
                    ty = r * ch
                    angle = random.uniform(0, math.pi * 2)
                    dist = random.uniform(500, 800)
                    sx = tx - img_w/2 + math.cos(angle) * dist
                    sy = ty - img_h/2 + math.sin(angle) * dist
                    delay = random.uniform(0, 0.4)
                    
                    self.chunks.append({
                        "sx": sx, "sy": sy,
                        "tx": tx, "ty": ty,
                        "cw": cw, "ch": ch,
                        "sc": c * cw, "sr": r * ch,
                        "delay": delay
                    })

        # Centre on screen
        screen = QDesktopWidget().screenGeometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2,
        )

        # Animation state
        self._t = 0.0               # master clock (incremented each tick)
        self._phase = 0             # 0=drawing lines, 1=glow, 2=loading
        self._line_progress = 0.0   # 0→1 overall line-draw progress
        self._glow_alpha = 0.0      # glow pulse opacity
        self._load_progress = 0.0   # 0→1 loading bar
        self._msg_index = 0
        self._particles = []
        self._finished = False

        # Generate ambient particles
        for _ in range(50):
            self._particles.append({
                "x": random.uniform(0, 720),
                "y": random.uniform(0, 620),
                "vx": random.uniform(-0.3, 0.3),
                "vy": random.uniform(-0.3, 0.3),
                "r": random.uniform(1, 3),
                "a": random.randint(30, 90),
            })

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(22)       # ~45 fps

    # ── Animation tick ──────────────────────────────────────────────

    def _tick(self):
        self._t += 0.022

        # Phase 0 — draw fox lines over ~2.5 s
        if self._phase == 0:
            self._line_progress = min(1.0, self._line_progress + 0.009)
            if self._line_progress >= 1.0:
                self._phase = 1
                self._glow_alpha = 0.0

        # Phase 1 — glow pulse for ~1 s
        elif self._phase == 1:
            self._glow_alpha = min(1.0, self._glow_alpha + 0.035)
            if self._glow_alpha >= 1.0:
                self._phase = 2
                self._load_progress = 0.0
                self._msg_index = 0

        # Phase 2 — loading bar over ~2.5 s
        elif self._phase == 2:
            self._load_progress = min(1.0, self._load_progress + 0.008)
            self._msg_index = min(
                len(self.LOADING_MESSAGES) - 1,
                int(self._load_progress * len(self.LOADING_MESSAGES)),
            )
            if self._load_progress >= 1.0 and not self._finished:
                self._finished = True
                QTimer.singleShot(400, self._emit_done)

        # Move particles
        for p in self._particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            if p["x"] < 0 or p["x"] > 720:
                p["vx"] *= -1
            if p["y"] < 0 or p["y"] > 620:
                p["vy"] *= -1

        self.update()

    def _emit_done(self):
        self._timer.stop()
        self.finished.emit()

    # ── Painting ────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # ── Background: rounded dark card with subtle gradient ──────
        bg_grad = QLinearGradient(0, 0, 0, h)
        bg_grad.setColorAt(0.0, QColor(0, 0, 0))
        bg_grad.setColorAt(1.0, QColor(0, 0, 0))
        p.setBrush(QBrush(bg_grad))
        p.setPen(QPen(QColor(COLORS["border"]), 2))
        p.drawRoundedRect(1, 1, w - 2, h - 2, 18, 18)

        # ── Ambient particles ───────────────────────────────────────
        for pt in self._particles:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(0, 201, 167, pt["a"])))
            p.drawEllipse(int(pt["x"]), int(pt["y"]), int(pt["r"]*2), int(pt["r"]*2))

        # ── Image Integration ────────────────────────────────────────────
        wolf_cx = w / 2
        wolf_cy = h / 2 - 60
        wolf_scale = min(w, h) * 0.42

        if hasattr(self, 'logo_pixmap') and self.logo_pixmap:
            logo_x = w/2 - self.logo_pixmap.width()/2
            logo_y = h/2 - 60 - self.logo_pixmap.height()/2
            
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            
            for chunk in self.chunks:
                t_local = max(0.0, min(1.0, (self._line_progress - chunk["delay"]) * 1.5))
                if t_local <= 0:
                    continue
                    
                eased = 1.0 - (1.0 - t_local)**3
                
                cx = chunk["sx"] + (chunk["tx"] - chunk["sx"]) * eased
                cy = chunk["sy"] + (chunk["ty"] - chunk["sy"]) * eased
                
                # Draw trailing rays tracking the moving block
                if t_local < 1.0:
                    ray_alpha = int(180 * (1.0 - t_local))
                    p.setPen(QPen(QColor(0, 229, 196, ray_alpha), 2))
                    p.drawLine(
                        int(logo_x + chunk["sx"] + chunk["cw"]/2), 
                        int(logo_y + chunk["sy"] + chunk["ch"]/2), 
                        int(logo_x + cx + chunk["cw"]/2), 
                        int(logo_y + cy + chunk["ch"]/2)
                    )
                
                # Draw the image chunk with fading opacity
                p.setOpacity(t_local)
                if t_local >= 1.0:
                    cx, cy = chunk["tx"], chunk["ty"] # Snap exactly if complete
                
                source_rect = QRectF(chunk["sc"], chunk["sr"], chunk["cw"], chunk["ch"])
                dest_rect = QRectF(logo_x + cx, logo_y + cy, chunk["cw"], chunk["ch"])
                p.drawPixmap(dest_rect, self.logo_pixmap, source_rect)
                
            p.setOpacity(1.0) # Reset transparency

        # ── Glow pulse ring around the fox (phase 1+) ───────────────
        if self._phase >= 1:
            pulse = 0.5 + 0.5 * math.sin(self._t * 3.0)
            ring_r = int(wolf_scale * 0.75 + pulse * 15)
            ring_a = int(40 * self._glow_alpha * (0.4 + 0.6 * pulse))
            pen_ring = QPen(QColor(0, 201, 167, ring_a), 2)
            p.setPen(pen_ring)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(int(wolf_cx) - ring_r, int(wolf_cy) - ring_r,
                          ring_r * 2, ring_r * 2)

        # ── App title ───────────────────────────────────────────────
        title_alpha = int(255 * min(1.0, self._line_progress * 2))
        p.setPen(QColor(0, 229, 196, title_alpha))
        p.setFont(QFont(FONT_FAMILY, 28, QFont.Bold))
        p.drawText(QRect(0, h - 170, w, 40), Qt.AlignCenter, "VIGILANT WOLF AI")

        p.setPen(QColor(122, 181, 168, int(title_alpha * 0.7)))
        p.setFont(QFont(FONT_FAMILY, 11))
        p.drawText(QRect(0, h - 135, w, 24), Qt.AlignCenter,
                   "Intelligent Driver Fatigue Detection")

        # ── Loading bar (phase 2) ───────────────────────────────────
        if self._phase >= 2:
            bar_y = h - 90
            bar_w = int(w * 0.6)
            bar_x = (w - bar_w) // 2
            bar_h = 6

            # Track
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(26, 58, 58)))
            p.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 3, 3)

            # Fill with gradient
            fill_w = max(1, int(bar_w * self._load_progress))
            bar_grad = QLinearGradient(bar_x, bar_y, bar_x + fill_w, bar_y)
            bar_grad.setColorAt(0.0, QColor(0, 201, 167))
            bar_grad.setColorAt(1.0, QColor(0, 229, 196))
            p.setBrush(QBrush(bar_grad))
            p.drawRoundedRect(bar_x, bar_y, fill_w, bar_h, 3, 3)

            # Shimmer highlight on fill
            shimmer_x = bar_x + int((self._t * 80) % fill_w)
            shimmer_grad = QRadialGradient(shimmer_x, bar_y + bar_h / 2, 30)
            shimmer_grad.setColorAt(0.0, QColor(255, 255, 255, 60))
            shimmer_grad.setColorAt(1.0, QColor(255, 255, 255, 0))
            p.setBrush(QBrush(shimmer_grad))
            p.drawRoundedRect(bar_x, bar_y, fill_w, bar_h, 3, 3)

            # Percentage text
            pct = int(self._load_progress * 100)
            p.setPen(QColor(0, 229, 196))
            p.setFont(QFont("Consolas", 10, QFont.Bold))
            p.drawText(QRect(bar_x + bar_w + 10, bar_y - 4, 50, 16),
                       Qt.AlignLeft | Qt.AlignVCenter, f"{pct}%")

            # Loading message
            p.setPen(QColor(122, 181, 168, 200))
            p.setFont(QFont(FONT_FAMILY, 10))
            msg = self.LOADING_MESSAGES[self._msg_index]
            p.drawText(QRect(0, bar_y + 16, w, 24), Qt.AlignCenter, msg)

        p.end()


# ═══════════════════════════════════════════════════════════════════════
#  Camera + Detection Thread
# ═══════════════════════════════════════════════════════════════════════

class DetectionThread(QThread):
    frame_ready  = pyqtSignal(np.ndarray, DetectionResult)
    camera_error = pyqtSignal(str)

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self._running = False
        self.engine = DetectionEngine()

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.camera_error.emit(f"Cannot open camera {self.camera_index}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue
            result = self.engine.process_frame(frame)
            self.frame_ready.emit(frame, result)
        cap.release()
        self.engine.release()

    def stop(self):
        self._running = False
        self.wait(3000)


# ═══════════════════════════════════════════════════════════════════════
#  Custom Widgets
# ═══════════════════════════════════════════════════════════════════════

class GaugeWidget(QWidget):
    """Modern circular gauge with gradient arc and glow effect."""

    def __init__(self, label="EAR", min_val=0.0, max_val=0.5, threshold=0.21,
                 color_normal=None, color_danger=None, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._label = label
        self._min   = min_val
        self._max   = max_val
        self._threshold    = threshold
        self._color_normal = QColor(color_normal or COLORS["eco_green"])
        self._color_danger = QColor(color_danger  or COLORS["alert_red"])
        self.setMinimumSize(140, 160)

    def set_value(self, v):
        self._value = max(self._min, min(self._max, v))
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2 - 8
        radius = min(w, h) // 2 - 20

        # Determine danger state
        is_danger = (self._value < self._threshold if self._label == "EAR"
                     else self._value > self._threshold)
        color = self._color_danger if is_danger else self._color_normal

        # Background card
        p.setPen(QPen(QColor(COLORS["border"]), 1))
        p.setBrush(QBrush(QColor(COLORS["surface_dark"])))
        p.drawRoundedRect(4, 4, w - 8, h - 8, 14, 14)

        # Track arc (subtle)
        track_pen = QPen(QColor(COLORS["surface_light"]), 7)
        track_pen.setCapStyle(Qt.RoundCap)
        p.setPen(track_pen)
        p.drawArc(cx - radius, cy - radius, radius * 2, radius * 2, 225 * 16, -270 * 16)

        # Value arc with glow
        ratio = (self._value - self._min) / (self._max - self._min + 1e-9)
        
        # Glow layer
        glow_color = QColor(color)
        glow_color.setAlpha(40)
        glow_pen = QPen(glow_color, 14)
        glow_pen.setCapStyle(Qt.RoundCap)
        p.setPen(glow_pen)
        p.drawArc(cx - radius, cy - radius, radius * 2, radius * 2, 225 * 16, int(-270 * ratio * 16))

        # Main arc
        arc_pen = QPen(color, 7)
        arc_pen.setCapStyle(Qt.RoundCap)
        p.setPen(arc_pen)
        p.drawArc(cx - radius, cy - radius, radius * 2, radius * 2, 225 * 16, int(-270 * ratio * 16))

        # Value text
        p.setPen(QColor(COLORS["text_primary"]))
        p.setFont(QFont(FONT_MONO, 20, QFont.Bold))
        p.drawText(QRect(0, cy - 16, w, 32), Qt.AlignCenter, f"{self._value:.2f}")

        # Label text
        p.setPen(QColor(COLORS["text_dim"]))
        p.setFont(QFont(FONT_FAMILY, 9, QFont.Bold))
        p.drawText(QRect(0, cy + radius - 2, w, 20), Qt.AlignCenter, self._label)

        # Status badge
        if self._label == "EAR":
            status = "CRITICAL" if self._value < self._threshold else "NORMAL"
        else:
            status = "YAWNING" if self._value > self._threshold else "NORMAL"
        
        badge_color = self._color_danger if is_danger else QColor(COLORS["success_green"])
        badge_bg = QColor(badge_color)
        badge_bg.setAlpha(25)
        
        badge_w, badge_h = 64, 18
        badge_x = cx - badge_w // 2
        badge_y = cy + radius + 14
        
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(badge_bg))
        p.drawRoundedRect(badge_x, badge_y, badge_w, badge_h, 4, 4)
        
        p.setPen(badge_color)
        p.setFont(QFont(FONT_FAMILY, 7, QFont.Bold))
        p.drawText(QRect(badge_x, badge_y, badge_w, badge_h), Qt.AlignCenter, status)
        
        p.end()


class StatusCard(QWidget):
    """Modern status metric card with accent strip and glass styling."""
    def __init__(self, icon_text, label, value="--", color=None, parent=None):
        super().__init__(parent)
        color = color or COLORS["eco_green"]
        self.setStyleSheet(f"""
            StatusCard {{
                background: {COLORS["surface_dark"]};
                border: 1px solid {COLORS["border"]};
                border-left: 3px solid {color};
                border-radius: 10px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)
        
        top = QHBoxLayout()
        icon = QLabel(icon_text)
        icon.setStyleSheet(f"color: {color}; font-size: 18px;")
        lbl  = QLabel(label)
        lbl.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px; "
                          "text-transform: uppercase; letter-spacing: 3px; font-weight: bold;")
        top.addWidget(icon)
        top.addWidget(lbl)
        top.addStretch()
        layout.addLayout(top)
        
        self._value_label = QLabel(value)
        self._value_label.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-size: 28px; font-weight: bold; "
            f"font-family: '{FONT_MONO}', monospace;")
        layout.addWidget(self._value_label)
        self._base_color = color
        add_shadow(self, blur_radius=15, y_offset=4, alpha=80)

    def set_value(self, v):
        self._value_label.setText(str(v))

    def set_color(self, c):
        self._value_label.setStyleSheet(
            f"color: {c}; font-size: 28px; font-weight: bold; "
            f"font-family: '{FONT_MONO}', monospace;")

class AlertBanner(QWidget):
    """Modern alert status banner with gradient accent and frosted glass."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(84)
        self._state  = DriverState.NORMAL
        self._reason = ""
        add_shadow(self, blur_radius=20, y_offset=6, alpha=100)

    def set_state(self, state: DriverState, reason: str):
        self._state  = state
        self._reason = reason
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        
        if self._state == DriverState.NORMAL:
            accent = COLORS["success_green"]
            title = "✓  ALL SYSTEMS NORMAL"
            bg_alpha, brd_alpha = 15, 50
        elif self._state == DriverState.WARNING:
            accent = COLORS["warning_orange"]
            title = "⚠  FATIGUE WARNING"
            bg_alpha, brd_alpha = 20, 60
        else:
            accent = COLORS["alert_red"]
            title = "🚨  CRITICAL ALERT"
            bg_alpha, brd_alpha = 30, 80

        # Card background
        bg = QColor(accent)
        bg.setAlpha(bg_alpha)
        brd = QColor(accent)
        brd.setAlpha(brd_alpha)
        p.setBrush(QBrush(bg))
        p.setPen(QPen(brd, 1))
        p.drawRoundedRect(2, 2, w - 4, h - 4, 12, 12)
        
        # Left accent strip with gradient
        accent_grad = QLinearGradient(6, 8, 6, h - 8)
        accent_grad.setColorAt(0.0, QColor(accent))
        accent_grad.setColorAt(1.0, QColor(COLORS.get("accent_emerald", accent)))
        p.setBrush(QBrush(accent_grad))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(6, 8, 4, h - 16, 2, 2)
        
        # Title
        p.setPen(QColor(COLORS["text_primary"]))
        p.setFont(QFont(FONT_FAMILY, 16, QFont.Bold))
        p.drawText(QRect(24, 8, w - 48, h // 2), Qt.AlignLeft | Qt.AlignVCenter, title)
        
        # Reason text
        if self._reason:
            p.setPen(QColor(accent))
            p.setFont(QFont(FONT_MONO, 13))
            p.drawText(QRect(24, h // 2, w - 48, h // 2 - 8), Qt.AlignLeft | Qt.AlignVCenter, self._reason)
        
        p.end()


# ═══════════════════════════════════════════════════════════════════════
#  Dashboard Window
# ═══════════════════════════════════════════════════════════════════════

class DashboardWindow(QMainWindow):
    trigger_alert       = pyqtSignal()
    dismiss_alert_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vigilant Wolf AI — Eco Drive Control")
        self.setMinimumSize(1220, 760)
        self._apply_dark_theme()

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(86)
        header.setStyleSheet(f"""
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #0f172a, stop:0.55 #020617, stop:1 #0f172a);
            border-bottom: 1px solid {COLORS['border']};
        """)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(24, 0, 24, 0)
        hl.setSpacing(16)

        logo_frame = QFrame()
        logo_frame.setFixedSize(56, 56)
        logo_frame.setStyleSheet(f"""
            background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 {COLORS['eco_green']}, stop:1 {COLORS['accent_emerald']});
            border-radius: 18px;
        """)
        logo_layout = QHBoxLayout(logo_frame)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        logo_label = QLabel("🐺")
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setStyleSheet("font-size: 28px; color: white;")
        logo_layout.addWidget(logo_label)
        hl.addWidget(logo_frame)

        title_stack = QVBoxLayout()
        title_stack.setSpacing(2)
        title_label = QLabel("Vigilant Wolf AI")
        title_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 22px; font-weight: bold; letter-spacing: 1px;")
        subtitle_label = QLabel("ECO MONITORING CONTROL ROOM")
        subtitle_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; letter-spacing: 2px; font-weight: bold;")
        title_stack.addWidget(title_label)
        title_stack.addWidget(subtitle_label)
        hl.addLayout(title_stack)
        hl.addStretch()

        live_chip = QFrame()
        live_chip.setStyleSheet(f"background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.18); border-radius: 18px;")
        live_chip_layout = QHBoxLayout(live_chip)
        live_chip_layout.setContentsMargins(16, 8, 16, 8)
        live_chip_layout.setSpacing(8)
        live_dot = QLabel("●")
        live_dot.setStyleSheet(f"color: {COLORS['eco_green']}; font-size: 14px;")
        live_state = QLabel("LIVE")
        live_state.setStyleSheet(f"color: {COLORS['eco_green']}; font-size: 13px; font-weight: bold; letter-spacing: 2px;")
        live_chip_layout.addWidget(live_dot)
        live_chip_layout.addWidget(live_state)
        hl.addWidget(live_chip)

        trip_frame = QFrame()
        trip_frame.setStyleSheet(f"background: {COLORS['surface_light']}; border-radius: 16px; border: 1px solid {COLORS['border']};")
        trip_fl = QHBoxLayout(trip_frame)
        trip_fl.setContentsMargins(16, 10, 16, 10)
        trip_lbl = QLabel("TRIP")
        trip_lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; letter-spacing: 2px; font-weight: bold;")
        self.trip_time_label = QLabel("00:00:00")
        self.trip_time_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 20px; font-weight: bold; font-family: 'Consolas', monospace;")
        trip_fl.addWidget(trip_lbl); trip_fl.addWidget(self.trip_time_label)
        hl.addWidget(trip_frame)

        self.status_dot  = QLabel("●")
        self.status_dot.setStyleSheet(f"color: {COLORS['success_green']}; font-size: 14px;")
        self.status_text = QLabel("SYSTEM ACTIVE")
        self.status_text.setStyleSheet(f"color: {COLORS['success_green']}; font-size: 13px; font-weight: bold; letter-spacing: 1.5px;")
        hl.addWidget(self.status_dot); hl.addWidget(self.status_text)

        self.btn_start = QPushButton("▶  START")
        self.btn_start.setStyleSheet(f"""
            QPushButton {{ background: rgba(32,217,160,0.15); color: {COLORS['success_green']};
                border: 1px solid rgba(32,217,160,0.35); border-radius: 6px;
                padding: 8px 20px; font-weight: bold; font-size: 14px; letter-spacing: 1px; }}
            QPushButton:hover {{ background: rgba(32,217,160,0.28); }}
        """)
        self.btn_stop = QPushButton("■  STOP")
        self.btn_stop.setStyleSheet(f"""
            QPushButton {{ background: rgba(255,71,87,0.10); color: {COLORS['alert_red']};
                border: 1px solid rgba(255,71,87,0.3); border-radius: 6px;
                padding: 8px 20px; font-weight: bold; font-size: 14px; letter-spacing: 1px; }}
            QPushButton:hover {{ background: rgba(255,71,87,0.22); }}
        """)
        hl.addWidget(self.btn_start); hl.addWidget(self.btn_stop)
        main_layout.addWidget(header)

        # ── Body ────────────────────────────────────────────────────
        body = QWidget()
        body.setStyleSheet(f"background: {COLORS['bg_dark']};")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(16, 16, 16, 16)
        body_layout.setSpacing(16)

        hero = QFrame()
        hero.setFixedHeight(90)
        hero.setStyleSheet(f"""
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 rgba(34, 197, 94, 0.15), stop:0.5 rgba(16, 185, 129, 0.08), stop:1 rgba(52, 211, 153, 0.12));
            border: 1px solid rgba(34, 197, 94, 0.3);
            border-radius: 20px;
        """)
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(24, 16, 24, 16)
        hero_layout.setSpacing(16)
        hero_copy = QVBoxLayout()
        hero_copy.setSpacing(6)
        hero_title = QLabel("AI COMMAND CENTER")
        hero_title.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 26px; font-weight: 900; letter-spacing: 2px;")
        hero_desc = QLabel("Real-time fatigue telemetry and predictive driver monitoring.")
        hero_desc.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 15px; letter-spacing: 0.5px;")
        hero_copy.addWidget(hero_title)
        hero_copy.addWidget(hero_desc)
        hero_layout.addLayout(hero_copy)
        hero_layout.addStretch()
        hero_badge = QLabel("ECO MODE ACTIVE")
        hero_badge.setStyleSheet(f"background: rgba(34,197,94,0.16); color: {COLORS['eco_green']}; border: 1px solid rgba(34,197,94,0.2); border-radius: 16px; padding: 10px 18px; font-size: 14px; font-weight: bold; letter-spacing: 2px;")
        hero_layout.addWidget(hero_badge)

        # Left: camera + EAR bar
        left_panel  = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        left_frame = QFrame()
        left_frame.setStyleSheet(f"background: {COLORS['surface_dark']}; border: 1px solid {COLORS['border']}; border-radius: 24px;")
        add_shadow(left_frame, blur_radius=40, y_offset=15, alpha=160)
        left_frame_layout = QVBoxLayout(left_frame)
        left_frame_layout.setContentsMargins(16, 16, 16, 16)
        left_frame_layout.setSpacing(16)

        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(420, 280)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("📷 INITIALIZING CAMERA...")
        self.camera_label.setStyleSheet(f"""
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid {COLORS['border_light']};
            border-radius: 16px; color: {COLORS['eco_green']}; font-size: 16px; font-weight: bold; letter-spacing: 2px;
        """)
        left_frame_layout.addWidget(self.camera_label, stretch=3)

        ear_bar_widget = QWidget()
        ear_bar_widget.setStyleSheet(f"background: {COLORS['surface_light']}; border-radius: 18px; border: 1px solid {COLORS['border']};")
        ear_bar_layout = QVBoxLayout(ear_bar_widget)
        ear_bar_layout.setContentsMargins(18, 14, 18, 14)
        ear_hdr = QHBoxLayout()
        ear_lbl = QLabel("EYE ASPECT RATIO (EAR)")
        ear_lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; letter-spacing: 2px; font-weight: bold;")
        self.ear_status_label = QLabel("NORMAL (0.30)")
        self.ear_status_label.setStyleSheet(f"color: {COLORS['success_green']}; font-size: 13px; font-weight: bold; letter-spacing: 1px;")
        ear_hdr.addWidget(ear_lbl); ear_hdr.addStretch(); ear_hdr.addWidget(self.ear_status_label)
        ear_bar_layout.addLayout(ear_hdr)

        self.ear_bar = QFrame()
        self.ear_bar.setFixedHeight(12)
        self.ear_bar.setStyleSheet(f"background: rgba(255,255,255,0.04); border-radius: 6px;")
        self.ear_bar_fill = QFrame(self.ear_bar)
        self.ear_bar_fill.setFixedHeight(12)
        self.ear_bar_fill.setStyleSheet(f"background: {COLORS['eco_green']}; border-radius: 6px;")
        ear_bar_layout.addWidget(self.ear_bar)
        left_frame_layout.addWidget(ear_bar_widget)
        left_layout.addWidget(left_frame)
        body_layout.addWidget(left_panel, stretch=35)

        # Centre: alert banner + gauges + cards
        center_panel  = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(12)

        center_frame = QFrame()
        center_frame.setStyleSheet(f"background: {COLORS['surface_dark']}; border: 1px solid {COLORS['border']}; border-radius: 24px;")
        add_shadow(center_frame, blur_radius=40, y_offset=15, alpha=160)
        center_frame_layout = QVBoxLayout(center_frame)
        center_frame_layout.setContentsMargins(16, 16, 16, 16)
        center_frame_layout.setSpacing(16)

        self.alert_banner = AlertBanner()
        center_frame_layout.addWidget(self.alert_banner)

        gauges_row = QHBoxLayout()
        self.ear_gauge = GaugeWidget("EAR", 0.0, 0.5, 0.21, COLORS["eco_teal"], COLORS["alert_red"])
        self.mar_gauge = GaugeWidget("MAR", 0.0, 1.0, 0.65, COLORS["eco_teal"], COLORS["warning_orange"])
        gauges_row.addWidget(self.ear_gauge); gauges_row.addWidget(self.mar_gauge)
        center_frame_layout.addLayout(gauges_row)
        center_frame_layout.addSpacing(10)

        stats_row = QGridLayout(); stats_row.setSpacing(12)
        self.perclos_card = StatusCard("👁", "PERCLOS",  "0%",     COLORS["eco_green"])
        self.yawn_card    = StatusCard("🥱", "Yawn",     "NO",     COLORS["eco_green"])
        self.head_card    = StatusCard("↔", "Head Pose","CENTER", COLORS["eco_green"])
        self.alerts_card  = StatusCard("🔔", "Alerts",  "0",      COLORS["alert_red"])
        stats_row.addWidget(self.perclos_card, 0, 0); stats_row.addWidget(self.yawn_card,   0, 1)
        stats_row.addWidget(self.head_card,   1, 0); stats_row.addWidget(self.alerts_card, 1, 1)
        center_frame_layout.addLayout(stats_row)
        center_layout.addWidget(center_frame)
        body_layout.addWidget(center_panel, stretch=40)

        # Right: safety score + SOS
        right_panel  = QWidget()
        right_panel.setStyleSheet(f"background: {COLORS['surface_dark']}; border-radius: 24px; border: 1px solid {COLORS['border']};")
        add_shadow(right_panel, blur_radius=40, y_offset=15, alpha=160)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(16)

        safety_title = QLabel("🛡  SAFETY SCORE")
        safety_title.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 18px; font-weight: bold; letter-spacing: 2px;")
        right_layout.addWidget(safety_title)

        self.safety_score_label = QLabel("A+")
        self.safety_score_label.setAlignment(Qt.AlignCenter)
        self.safety_score_label.setStyleSheet(f"color: {COLORS['eco_green']}; font-size: 84px; font-weight: 900;")
        right_layout.addWidget(self.safety_score_label)

        score_copy = QLabel("ECO-FOCUSED SAFETY INDEX")
        score_copy.setAlignment(Qt.AlignCenter)
        score_copy.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; letter-spacing: 2px; font-weight: bold;")
        right_layout.addWidget(score_copy)

        conf_title = QLabel("DANGER CONFIDENCE")
        conf_title.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; letter-spacing: 2px; font-weight: bold; margin-top: 20px;")
        right_layout.addWidget(conf_title)
        self.confidence_label = QLabel("0%")
        self.confidence_label.setStyleSheet(f"color: {COLORS['eco_green']}; font-size: 38px; font-weight: bold;")
        right_layout.addWidget(self.confidence_label)
        right_layout.addStretch()

        self.btn_sos = QPushButton("🚨  EMERGENCY SOS")
        self.btn_sos.setFixedHeight(72)
        self.btn_sos.setStyleSheet(f"""
            QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {COLORS['alert_red']}, stop:1 #ff7b84); color: white;
                border: none; border-radius: 18px;
                font-size: 24px; font-weight: 900; letter-spacing: 3px; }}
            QPushButton:hover {{ background: #ff6a75; }}
        """)
        self.btn_sos.clicked.connect(lambda: webbrowser.open(f"tel:{EMERGENCY_NUMBER}"))
        right_layout.addWidget(self.btn_sos)
        body_layout.addWidget(right_panel, stretch=20)
        main_layout.addWidget(body, stretch=1)

        # ── Footer ──────────────────────────────────────────────────
        self.footer = QWidget()
        self.footer.setFixedHeight(64)
        self.footer.setStyleSheet(f"background: {COLORS['surface_dark']}; border-top: 1px solid {COLORS['border']};")
        fl = QHBoxLayout(self.footer)
        fl.setContentsMargins(24, 0, 24, 0)
        self.footer_status = QLabel("● SYSTEM READY")
        self.footer_status.setStyleSheet(f"color: {COLORS['eco_green']}; font-size: 14px; font-weight: bold; letter-spacing: 2px;")
        fl.addWidget(self.footer_status); fl.addStretch()
        self.footer_fps = QLabel("FPS: --")
        self.footer_fps.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 13px; font-family: Consolas, monospace; letter-spacing: 1.5px;")
        fl.addWidget(self.footer_fps)
        main_layout.addWidget(self.footer)

        # ── State ───────────────────────────────────────────────────
        self._detection_thread: DetectionThread | None = None
        self._trip_start    = None
        self._alert_count   = 0
        self._was_critical  = False
        self._frame_times   = []
        self._last_frame_time = time.time()

        self._trip_timer = QTimer()
        self._trip_timer.timeout.connect(self._update_trip_time)
        self._trip_timer.start(1000)

        self.btn_start.clicked.connect(self.start_detection)
        self.btn_stop.clicked.connect(self.stop_detection)
        QTimer.singleShot(500, self.start_detection)

    def _apply_dark_theme(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background: {COLORS['bg_dark']}; }}
            QWidget  {{ color: {COLORS['text_primary']}; font-family: '{FONT_FAMILY}'; }}
            QLabel   {{ background: transparent; border: none; }}
        """)

    def start_detection(self):
        if self._detection_thread and self._detection_thread.isRunning():
            return
        self._trip_start = time.time()
        self._detection_thread = DetectionThread(camera_index=0)
        self._detection_thread.frame_ready.connect(self._on_frame)
        self._detection_thread.camera_error.connect(self._on_camera_error)
        self._detection_thread.start()
        self.footer_status.setText("● DETECTION ACTIVE")
        self.footer_status.setStyleSheet(f"color: {COLORS['success_green']}; font-size: 11px; font-weight: bold;")

    def stop_detection(self):
        if self._detection_thread:
            self._detection_thread.stop()
            self._detection_thread = None
        self.footer_status.setText("● SYSTEM STOPPED")
        self.footer_status.setStyleSheet(f"color: {COLORS['warning_orange']}; font-size: 11px; font-weight: bold;")

    def _on_camera_error(self, msg):
        self.camera_label.setText(f"❌ {msg}")

    def _on_frame(self, frame: np.ndarray, result: DetectionResult):
        # FPS
        now = time.time()
        dt  = now - self._last_frame_time
        self._last_frame_time = now
        self._frame_times.append(dt)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        avg_dt = sum(self._frame_times) / len(self._frame_times) if self._frame_times else 1
        self.footer_fps.setText(f"FPS: {1/avg_dt:.0f}" if avg_dt > 0 else "FPS: --")

        # Overlays on frame
        display_frame = frame.copy()
        h, w, _ = display_frame.shape
        if result.face_detected and result.face_bbox:
            bx, by, bw, bh = result.face_bbox
            if result.state == DriverState.CRITICAL:
                rect_color = (57, 71, 255)
            elif result.state == DriverState.WARNING:
                rect_color = (2, 165, 255)
            else:
                rect_color = (167, 217, 32)
            cv2.rectangle(display_frame, (bx, by), (bx+bw, by+bh), rect_color, 2)
            from detection_engine import LEFT_EYE, RIGHT_EYE
            if result.landmarks_2d:
                for idx in LEFT_EYE + RIGHT_EYE:
                    pt = result.landmarks_2d[idx]
                    cv2.circle(display_frame, pt, 2, (0, 201, 167), -1)
            label = result.alert_reason[:40] if result.state != DriverState.NORMAL and result.alert_reason else result.state.value
            cv2.putText(display_frame, label, (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rect_color, 2)

        rgb  = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888)
        scaled = QPixmap.fromImage(qimg).scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label.setPixmap(scaled)

        # Gauges
        self.ear_gauge.set_value(result.ear)
        self.mar_gauge.set_value(result.mar)

        # EAR bar
        bar_w  = self.ear_bar.width()
        fill_w = int(bar_w * min(1.0, result.ear / 0.5))
        self.ear_bar_fill.setFixedWidth(max(1, fill_w))
        if result.ear < 0.21:
            self.ear_bar_fill.setStyleSheet(f"background: {COLORS['alert_red']}; border-radius: 4px;")
            self.ear_status_label.setText(f"CRITICAL ({result.ear:.2f})")
            self.ear_status_label.setStyleSheet(f"color: {COLORS['alert_red']}; font-size: 11px; font-weight: bold;")
        elif result.ear < 0.25:
            self.ear_bar_fill.setStyleSheet(f"background: {COLORS['warning_orange']}; border-radius: 4px;")
            self.ear_status_label.setText(f"WARNING ({result.ear:.2f})")
            self.ear_status_label.setStyleSheet(f"color: {COLORS['warning_orange']}; font-size: 11px; font-weight: bold;")
        else:
            self.ear_bar_fill.setStyleSheet(f"background: {COLORS['success_green']}; border-radius: 4px;")
            self.ear_status_label.setText(f"NORMAL ({result.ear:.2f})")
            self.ear_status_label.setStyleSheet(f"color: {COLORS['success_green']}; font-size: 11px; font-weight: bold;")

        # Cards
        self.perclos_card.set_value(f"{result.perclos*100:.0f}%")
        self.perclos_card.set_color(COLORS['alert_red'] if result.perclos > 0.4 else COLORS['eco_green'])
        self.yawn_card.set_value("YES" if result.is_yawning else "NO")
        self.yawn_card.set_color(COLORS['warning_orange'] if result.is_yawning else COLORS['success_green'])
        head_text = "CENTER"
        if result.is_head_turned: head_text = f"TURNED {result.yaw:.0f}°"
        if result.is_nodding:     head_text = f"NODDING {result.pitch:.0f}°"
        self.head_card.set_value(head_text)
        self.head_card.set_color(COLORS['alert_red'] if (result.is_head_turned or result.is_nodding) else COLORS['success_green'])
        self.confidence_label.setText(f"{result.confidence*100:.0f}%")
        self.confidence_label.setStyleSheet(
            f"color: {COLORS['alert_red'] if result.confidence > 0.5 else COLORS['eco_green']}; font-size: 22px; font-weight: bold;")

        # Alert banner
        self.alert_banner.set_state(result.state, result.alert_reason)

        # Safety score
        if result.perclos < 0.1 and result.confidence < 0.1:
            score, color = "A+", COLORS['success_green']
        elif result.perclos < 0.2 and result.confidence < 0.3:
            score, color = "A",  COLORS['success_green']
        elif result.perclos < 0.3:
            score, color = "B+", COLORS['eco_green']
        elif result.perclos < 0.4:
            score, color = "B",  COLORS['warning_orange']
        else:
            score, color = "C",  COLORS['alert_red']
        self.safety_score_label.setText(score)
        self.safety_score_label.setStyleSheet(f"color: {color}; font-size: 48px; font-weight: bold;")

        # Footer
        if result.state == DriverState.CRITICAL:
            self.footer.setStyleSheet(f"background: {COLORS['alert_red']}; border-top: none;")
            self.footer_status.setText(f"⚠ WAKE UP! — {result.alert_reason}")
            self.footer_status.setStyleSheet("color: white; font-size: 12px; font-weight: bold;")
        elif result.state == DriverState.WARNING:
            self.footer.setStyleSheet(f"background: {COLORS['warning_orange']}; border-top: none;")
            self.footer_status.setText(f"⚠ CAUTION — {result.alert_reason}")
            self.footer_status.setStyleSheet("color: white; font-size: 12px; font-weight: bold;")
        else:
            self.footer.setStyleSheet(f"background: {COLORS['surface_dark']}; border-top: 1px solid {COLORS['border']};")
            self.footer_status.setText("● DETECTION ACTIVE")
            self.footer_status.setStyleSheet(f"color: {COLORS['success_green']}; font-size: 11px; font-weight: bold;")

        # Trigger alert
        if result.state == DriverState.CRITICAL and not self._was_critical:
            self._was_critical = True
            self._alert_count += 1
            self.alerts_card.set_value(str(self._alert_count))
            self.trigger_alert.emit()
        elif result.state != DriverState.CRITICAL:
            self._was_critical = False

    def _update_trip_time(self):
        if self._trip_start:
            e = int(time.time() - self._trip_start)
            self.trip_time_label.setText(f"{e//3600:02d}:{(e%3600)//60:02d}:{e%60:02d}")

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════
#  3-D Animated Danger Canvas
# ═══════════════════════════════════════════════════════════════════════

class DangerCanvas(QWidget):
    """
    Animated danger background:
    - Radial speed-lines rushing outward from centre (red, glowing)
    - Depth-fade particles (dots growing as they rush outward)
    - Rotating danger hexagon drawn with QPainter
    Repaints every 30 ms for ~33 fps animation.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._t      = 0.0     # animation clock
        self._lines  = self._gen_lines(42)
        self._sparks = self._gen_sparks(60)
        self._timer  = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def _gen_lines(self, n):
        """Each line: (angle_deg, speed, phase_offset, alpha_base)"""
        return [(random.uniform(0, 360), random.uniform(0.6, 1.4),
                 random.uniform(0, 1.0), random.randint(120, 220)) for _ in range(n)]

    def _gen_sparks(self, n):
        """Each spark: (angle_deg, speed, phase, size_max)"""
        return [(random.uniform(0, 360), random.uniform(0.4, 1.2),
                 random.uniform(0, 1.0), random.randint(3, 8)) for _ in range(n)]

    def start(self):  self._timer.start(30)
    def stop(self):   self._timer.stop()

    def _tick(self):
        self._t += 0.08  # Increased speed for more urgency
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        max_r   = math.hypot(cx, cy)

        # Black-red radial gradient background
        grad = QRadialGradient(cx, cy, max_r)
        grad.setColorAt(0.0, QColor(80, 5, 5))
        grad.setColorAt(0.5, QColor(30, 2, 2))
        grad.setColorAt(1.0, QColor(8,  0, 0))
        p.fillRect(0, 0, w, h, grad)

        # ── Speed lines ──────────────────────────────────────────────
        for angle_deg, speed, phase, alpha_base in self._lines:
            phase_val = ((self._t * speed * 1.5 + phase) % 1.0) # Faster rush
            r_start   = max_r * phase_val * 0.12
            r_end     = max_r * min(1.0, phase_val + 0.22)
            alpha     = int(alpha_base * (1.0 - phase_val * 0.6))
            rad       = math.radians(angle_deg)
            x1 = cx + r_start * math.cos(rad)
            y1 = cy + r_start * math.sin(rad)
            x2 = cx + r_end   * math.cos(rad)
            y2 = cy + r_end   * math.sin(rad)

            # Glow: wide semi-transparent + thin bright
            pen = QPen(QColor(255, 60, 70, max(0, alpha // 3)), 5)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(int(x1), int(y1), int(x2), int(y2))

            pen2 = QPen(QColor(255, 100, 110, alpha), 2)
            pen2.setCapStyle(Qt.RoundCap)
            p.setPen(pen2)
            p.drawLine(int(x1), int(y1), int(x2), int(y2))

        # ── Depth particles ──────────────────────────────────────────
        for angle_deg, speed, phase, sz_max in self._sparks:
            phase_val = ((self._t * speed * 0.7 + phase) % 1.0)
            r   = max_r * 0.9 * phase_val
            rad = math.radians(angle_deg)
            sx  = cx + r * math.cos(rad)
            sy  = cy + r * math.sin(rad)
            sz  = int(sz_max * phase_val)
            a   = int(200 * (1.0 - phase_val))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(255, 80, 90, a)))
            p.drawEllipse(int(sx - sz/2), int(sy - sz/2), sz, sz)

        # ── Rotating danger hexagon ──────────────────────────────────
        rot = self._t * 25  # degrees per tick
        hex_r = min(w, h) * 0.13
        p.save()
        p.translate(cx, cy)
        p.rotate(rot)
        hex_pen = QPen(QColor(255, 71, 87, 160), 4)
        p.setPen(hex_pen)
        p.setBrush(Qt.NoBrush)
        pts = []
        for i in range(6):
            a2 = math.radians(60 * i)
            pts.append(QPointF(hex_r * math.cos(a2), hex_r * math.sin(a2)))
        for i in range(6):
            p.drawLine(pts[i], pts[(i+1) % 6])
        # Inner warning symbol (exclamation)
        p.setPen(QPen(QColor(255, 71, 87, 200), 5))
        p.drawLine(QPointF(0, -hex_r*0.45), QPointF(0, hex_r*0.10))
        p.drawPoint(QPointF(0, hex_r*0.32))
        p.restore()

        p.end()


# ═══════════════════════════════════════════════════════════════════════
#  Full-Screen Alert Window
# ═══════════════════════════════════════════════════════════════════════

class AlertWindow(QMainWindow):
    """
    Full-screen red alert:
    - Animated danger canvas (speed lines + particles + hex icon)
    - Large centred "WAKE UP!" message
    - Dual audio: rapid beep sequence → TTS spoken warning
    - 10-second countdown → auto-call 9650427590
    """

    dismissed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CRITICAL ALERT")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)

        # Container
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(80)
        header.setStyleSheet("""
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #1c0404, stop:1 #2e0808);
            border-bottom: 4px solid #ff4757;
        """)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(24, 0, 24, 0)
        warn_icon = QLabel("⚠")
        warn_icon.setStyleSheet("font-size: 42px;")
        title_lbl = QLabel("CRITICAL ALERT")
        title_lbl.setStyleSheet("color: #ff4757; font-size: 28px; font-weight: bold; letter-spacing: 3px;")
        sub_lbl = QLabel("FATIGUE INTERVENTION MODE")
        sub_lbl.setStyleSheet("color: #ffcccc; font-size: 13px; font-weight: bold; letter-spacing: 2px;")
        vtitles = QVBoxLayout()
        vtitles.addWidget(title_lbl); vtitles.addWidget(sub_lbl)
        hl.addWidget(warn_icon); hl.addLayout(vtitles); hl.addStretch()
        badge = QLabel("DROWSINESS\nCRITICAL")
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet("background:#ff4757;color:white;font-size:15px;font-weight:bold;border-radius:8px;padding:8px 16px;")
        hl.addWidget(badge)
        root_layout.addWidget(header)

        # ── Main body: animated canvas + overlay text ────────────────
        body_container = QWidget()
        body_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        stacked = QStackedLayout(body_container)
        stacked.setStackingMode(QStackedLayout.StackAll)

        # Layer 0: animated danger canvas (background)
        self._canvas = DangerCanvas()
        stacked.addWidget(self._canvas)

        # Layer 1: text overlay (transparent background)
        overlay = QWidget()
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        overlay.setStyleSheet("background: transparent;")
        ov_layout = QVBoxLayout(overlay)
        ov_layout.setAlignment(Qt.AlignCenter)

        self.wake_label = QLabel("WAKE  UP!")
        self.wake_label.setAlignment(Qt.AlignCenter)
        self.wake_label.setStyleSheet(
            "color: #ff4757; font-size: 118px; font-weight: 900; letter-spacing: -2px; background: transparent;")
        ov_layout.addWidget(self.wake_label)

        divider = QFrame()
        divider.setFixedSize(140, 5)
        divider.setStyleSheet("background: #ff4757; border-radius: 3px;")
        ov_layout.addWidget(divider, alignment=Qt.AlignCenter)

        pull_over = QLabel("PULL OVER IMMEDIATELY")
        pull_over.setAlignment(Qt.AlignCenter)
        pull_over.setStyleSheet(
            "color: #ffffff; font-size: 38px; font-weight: bold; letter-spacing: 4px; background: transparent; margin-top: 10px;")
        ov_layout.addWidget(pull_over)

        desc = QLabel("Critical fatigue levels detected. Driving is no longer safe.\nFind a safe stopping point now.")
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color: #cc8888; font-size: 15px; background: transparent; margin-top: 8px;")
        ov_layout.addWidget(desc)

        ov_layout.addSpacing(16)

        self.reason_label = QLabel("")
        self.reason_label.setAlignment(Qt.AlignCenter)
        self.reason_label.setStyleSheet(
            "color: #ff8090; font-size: 17px; font-weight: bold; font-family: Consolas, monospace; background: transparent;")
        ov_layout.addWidget(self.reason_label)

        ov_layout.addSpacing(16)

        # Countdown label
        self.countdown_label = QLabel(f"📞 Auto-calling {EMERGENCY_NUMBER} in 10 s…")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet(
            "color: #ffaa00; font-size: 20px; font-weight: bold; background: transparent;")
        ov_layout.addWidget(self.countdown_label)

        stacked.addWidget(overlay)
        stacked.setCurrentIndex(1)
        root_layout.addWidget(body_container, stretch=1)

        # ── Action buttons ───────────────────────────────────────────
        btn_area = QWidget()
        btn_area.setFixedHeight(120)
        btn_area.setStyleSheet("background: #160404;")
        btn_layout = QHBoxLayout(btn_area)
        btn_layout.setContentsMargins(40, 14, 40, 14)
        btn_layout.setSpacing(20)

        self.btn_awake = QPushButton("✓  I AM AWAKE")
        self.btn_awake.setFixedHeight(86)
        self.btn_awake.setStyleSheet("""
            QPushButton { background: white; color: #222;
                border: 4px solid #ccc; border-radius: 14px;
                font-size: 24px; font-weight: 900; }
            QPushButton:hover { background: #e8fff5; border-color: #20d9a0; }
            QPushButton:pressed { background: #20d9a0; color: white; }
        """)
        self.btn_awake.clicked.connect(self.dismiss)

        self.btn_mute = QPushButton("🔇  MUTE ALARM")
        self.btn_mute.setFixedHeight(86)
        self.btn_mute.setStyleSheet("""
            QPushButton { background: white; color: #222;
                border: 4px solid #ffa502; border-radius: 14px;
                font-size: 22px; font-weight: 900; }
            QPushButton:hover { background: #fff3e0; }
        """)
        self.btn_mute.clicked.connect(self._mute_alarm)

        self.btn_call = QPushButton(f"📞  CALL {EMERGENCY_NUMBER}")
        self.btn_call.setFixedHeight(86)
        self.btn_call.setStyleSheet("""
            QPushButton { background: #ff4757; color: white;
                border: 4px solid #ff6b6b; border-radius: 14px;
                font-size: 20px; font-weight: 900; letter-spacing: 1px; }
            QPushButton:hover { background: #ff6b6b; }
        """)
        self.btn_call.clicked.connect(self._manual_call)

        btn_layout.addWidget(self.btn_awake)
        btn_layout.addWidget(self.btn_mute)
        btn_layout.addWidget(self.btn_call)
        root_layout.addWidget(btn_area)

        # ── Footer ───────────────────────────────────────────────────
        footer = QWidget()
        footer.setFixedHeight(36)
        footer.setStyleSheet("background: #2b0808; border-top: 1px solid rgba(255,71,87,0.3);")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(16, 0, 16, 0)
        fl.addWidget(QLabel("● LIVE MONITORING ACTIVE", styleSheet="color:#ff4757;font-size:11px;font-weight:bold;font-family:Consolas,monospace;"))
        fl.addStretch()
        root_layout.addWidget(footer)

        # ── Internal state ───────────────────────────────────────────
        self._alarm_muted  = False
        self._alarm_player = None
        self._tts_thread   = None

        # Pulse animation for wake_label
        self._pulse_t   = 0.0
        self._pulse_dir = 1
        self._pulse_timer = QTimer()
        self._pulse_timer.timeout.connect(self._pulse_effect)

        # Auto-call countdown
        self._countdown_val = 10
        self._countdown_timer = QTimer()
        self._countdown_timer.timeout.connect(self._countdown_tick)
        self._called = False

    # ── Public API ──────────────────────────────────────────────────

    def show_alert(self, reason: str = ""):
        self.reason_label.setText(reason)
        screen = QDesktopWidget().screenGeometry()
        self.setGeometry(screen)
        self.showFullScreen()

        self._canvas.start()
        self._pulse_timer.start(50)
        self._start_alarm()

        # Reset and start countdown
        self._called = False
        self._countdown_val = 10
        self.countdown_label.setText(f"📞 Auto-calling {EMERGENCY_NUMBER} in {self._countdown_val} s…")
        self.countdown_label.setStyleSheet(
            "color: #ffaa00; font-size: 20px; font-weight: bold; background: transparent;")
        self.btn_mute.setText("🔇  MUTE ALARM")
        self.btn_mute.setEnabled(True)
        self._countdown_timer.start(1000)

    def dismiss(self):
        self._countdown_timer.stop()
        self._pulse_timer.stop()
        self._canvas.stop()
        self._stop_alarm()
        self.hide()
        self.dismissed.emit()

    # ── Countdown & Auto-Call ────────────────────────────────────────

    def _countdown_tick(self):
        self._countdown_val -= 1
        if self._countdown_val > 0:
            self.countdown_label.setText(
                f"📞 Auto-calling {EMERGENCY_NUMBER} in {self._countdown_val} s…")
        else:
            self._countdown_timer.stop()
            if not self._called:
                self._called = True
                self._do_call()

    def _do_call(self):
        self.countdown_label.setText(f"📞 Calling {EMERGENCY_NUMBER}…  Connecting…")
        self.countdown_label.setStyleSheet(
            "color: #ff4757; font-size: 22px; font-weight: bold; background: transparent;")
        try:
            webbrowser.open(f"tel:{EMERGENCY_NUMBER}")
        except Exception:
            pass

    def _manual_call(self):
        self._countdown_timer.stop()
        self._called = True
        self._do_call()

    # ── Alarm (beep + TTS) ───────────────────────────────────────────

    def _mute_alarm(self):
        self._alarm_muted = True
        self._stop_alarm()
        self.btn_mute.setText("🔇  MUTED")
        self.btn_mute.setEnabled(False)

    def _start_alarm(self):
        if self._alarm_muted:
            return

        audio_path = os.path.join(os.path.dirname(__file__), 'assets', 'alert.wav')
        if HAS_MULTIMEDIA and os.path.exists(audio_path):
            try:
                self._alarm_player = QMediaPlayer()
                url = QUrl.fromLocalFile(os.path.abspath(audio_path))
                self._alarm_player.setMedia(QMediaContent(url))
                self._alarm_player.setVolume(100)
                self._alarm_player.mediaStatusChanged.connect(self._on_alarm_status)
                self._alarm_player.play()
            except Exception:
                pass

        # Dual-cue: beep sequence + spoken warning in background thread
        def _dual_cue():
            # Stage 1: rapid 4-tone beep sequence (×2)
            for _ in range(2):
                if self._alarm_muted: return
                if SYSTEM == "Windows":
                    import winsound
                    winsound.Beep(880,  250)
                    winsound.Beep(660,  250)
                    winsound.Beep(880,  250)
                    winsound.Beep(1100, 400)
                else:
                    print('\a', end='', flush=True)
                time.sleep(0.1)

            if self._alarm_muted: return
            time.sleep(0.5)

            # Stage 2: TTS spoken warning
            if HAS_TTS:
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 155)
                    engine.setProperty('volume', 1.0)
                    engine.say(
                        "Warning! Driver fatigue detected. "
                        "Please pull over immediately and rest."
                    )
                    engine.runAndWait()
                    engine.stop()
                except Exception:
                    pass
            else:
                # Fallback: second beep round if no TTS
                if SYSTEM == "Windows":
                    import winsound
                    for _ in range(3):
                        if self._alarm_muted: return
                        winsound.Beep(1200, 300)
                        time.sleep(0.15)

        self._tts_thread = threading.Thread(target=_dual_cue, daemon=True)
        self._tts_thread.start()

    def _on_alarm_status(self, status):
        if HAS_MULTIMEDIA and status == QMediaPlayer.EndOfMedia:
            if self.isVisible() and not self._alarm_muted:
                self._alarm_player.setPosition(0)
                self._alarm_player.play()

    def _stop_alarm(self):
        if self._alarm_player:
            try:
                self._alarm_player.stop()
            except Exception:
                pass

    # ── Pulse wake_label ─────────────────────────────────────────────

    def _pulse_effect(self):
        self._pulse_t += 0.06 * self._pulse_dir
        if self._pulse_t >= 1.0:
            self._pulse_dir = -1
        elif self._pulse_t <= 0.0:
            self._pulse_dir = 1
        sz = 118 + int(10 * self._pulse_t)
        self.wake_label.setStyleSheet(
            f"color: #ff4757; font-size: {sz}px; font-weight: 900; letter-spacing: -2px; background: transparent;")


# ═══════════════════════════════════════════════════════════════════════
#  Application Entry Point
# ═══════════════════════════════════════════════════════════════════════

class VigilantWolfApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("Vigilant Wolf AI")

        self.splash = StartupWolfWindow()
        self.dashboard = None
        self.alert_window = None

    def _on_splash_done(self):
        self.splash.close()
        self.dashboard    = DashboardWindow()
        self.alert_window = AlertWindow()
        self.dashboard.trigger_alert.connect(self._show_alert)
        self.alert_window.dismissed.connect(self._dismiss_alert)
        self.dashboard.show()

    def _show_alert(self):
        reason = self.dashboard.alert_banner._reason
        self.alert_window.show_alert(reason)

    def _dismiss_alert(self):
        self.alert_window._alarm_muted = False
        if self.dashboard._detection_thread:
            self.dashboard._detection_thread.engine.reset()

    def run(self):
        self.splash.finished.connect(self._on_splash_done)
        self.splash.show()
        return self.app.exec_()


if __name__ == "__main__":
    app = VigilantWolfApp()
    sys.exit(app.run())
