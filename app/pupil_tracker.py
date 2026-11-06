"""
Pupil Cam2 Binocular Eye Tracker v1.0 – direct UVC, no services
================================================================
Adapted specifically for the two Pupil Cam2 ID0 sensors in this headset:
USB VID:PID 0C45:64AB, libusbK, dual 400x400 @ 30 fps MJPG.

Uses pupil-labs-uvc to open Eye 0 and Eye 1 directly via libusbK.  Keeps
the original Detector2D + pye3d eye-tracking pipeline, ROI selection,
exposure controls, gaze view, and JSON settings.  No Pupil Service, no ZMQ.

Version 1.0 retains the v0.2 UI and pipeline, adds native evidence-guided
reacquisition for side-view pupils and compact validity-aware CSV export.
No predicted measurements or uncalibrated screen coordinates are exported.

Requirements (Python 3.10):
    pip install pupil-labs-uvc pupil-detectors pye3d opencv-python Pillow sv-ttk numpy"<2"

Run:
    Use ../02_start.bat (bundled isolated Python runtime).
"""

from __future__ import annotations

import contextlib
import collections
import io
import json
import math
import os
import pathlib
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox
from recording import SessionRecorder

import cv2
import numpy as np
import uvc
from PIL import Image, ImageTk
from pupil_detectors import Detector2D, Roi
from pye3d.detector_3d import CameraModel, Detector3D


@contextlib.contextmanager
def _suppress_stdout():
    """Suppress Python-level print() output (e.g. pye3d observation warnings)."""
    import sys
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old

try:
    import sv_ttk
    _HAS_SV_TTK = True
except ImportError:
    _HAS_SV_TTK = False

# ── constants ──────────────────────────────────────────────────────────────────

PREVIEW_W = 400
PREVIEW_H = 400

APP_VERSION = "1.0"

# This Cam2 firmware advertises MJPG for all supported modes.
PREFERRED_FORMATS = ("MJPG", "GREY", "Y800", "BY8 ")

TARGET_W   = 400
TARGET_H   = 400
TARGET_FPS = 30

# This Cam2 firmware exposes the UVC selector but rejects SET_CUR=Auto.
CAM2_AUTO_EXPOSURE_SUPPORTED = False

# Pupil Labs' published default intrinsics for Pupil Cam2 eye cameras.
# pye3d accepts one focal-length scalar, so use the mean of fx and fy.
# 400x400: fx=561.471804, fy=562.494105
# 192x192: fx=282.976877, fy=283.561467
CAM2_FOCAL_LENGTHS = {
    (400, 400): (561.471804 + 562.494105) / 2.0,
    (192, 192): (282.976877 + 283.561467) / 2.0,
}

COLORMAPS = {
    "Grayscale": None,
    "Inferno":   cv2.COLORMAP_INFERNO,
    "Hot":       cv2.COLORMAP_HOT,
    "Bone":      cv2.COLORMAP_BONE,
    "Viridis":   cv2.COLORMAP_VIRIDIS,
}

EYE_LABELS = ["Eye 0", "Eye 1"]

APP_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_ROOT = APP_ROOT / "data"
SETTINGS_PATH = DATA_ROOT / "settings.json"
LEGACY_SETTINGS_PATH = SETTINGS_PATH
DEBUG_SAVE_DIR = DATA_ROOT / "debug"


def _cam2_focal_length(width: int, height: int) -> float:
    """Return the official Cam2 focal length, scaled for an unknown mode."""
    exact = CAM2_FOCAL_LENGTHS.get((int(width), int(height)))
    if exact is not None:
        return exact
    # Cam2 modes are square. Scaling the measured 400x400 intrinsics is a much
    # safer fallback than the former 0.44*width dummy value.
    return CAM2_FOCAL_LENGTHS[(400, 400)] * (float(width) / 400.0)


def _pye3d_direction(theta, phi):
    """pye3d geometry.utilities.sph2cart: x=sin(t)cos(p), y=cos(t), z=sin(t)sin(p)."""
    return (math.sin(theta) * math.cos(phi), math.cos(theta),
            math.sin(theta) * math.sin(phi))


@dataclass
class TrackingDecision:
    ellipse: Optional[dict]
    source: str                 # PRIMARY, ASSIST, PREDICT, LOST
    confidence: float
    primary_accepted: bool
    reason: str
    contrast: float = 0.0


def _normalise_ellipse(ellipse: Optional[dict]) -> Optional[dict]:
    """Return a finite OpenCV-style ellipse with minor axis first."""
    if not ellipse:
        return None
    try:
        cx, cy = (float(v) for v in ellipse["center"])
        axis0, axis1 = (float(v) for v in ellipse["axes"])
        angle = float(ellipse["angle"]) % 180.0
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    values = np.asarray([cx, cy, axis0, axis1, angle], dtype=np.float64)
    if not np.all(np.isfinite(values)) or axis0 <= 0.0 or axis1 <= 0.0:
        return None
    if axis0 > axis1:
        axis0, axis1 = axis1, axis0
        angle = (angle + 90.0) % 180.0
    return {"center": (cx, cy), "axes": (axis0, axis1), "angle": angle}


class _TemporalPupilBase:
    """Reject isolated false ellipses and bridge short Detector2D dropouts.

    The auxiliary contour detector is deliberately subordinate to Detector2D:
    its confidence is capped below pye3d's model-update thresholds. It can keep
    an already-built 3D model alive, but cannot train or poison that model.
    """

    MAX_PREDICT_FRAMES = 4
    MAX_STATE_MISSES = 18

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._state: Optional[np.ndarray] = None  # cx, cy, minor_d, major_d, angle
        self._velocity = np.zeros(2, dtype=np.float64)
        self._last_measurement_ts: Optional[float] = None
        self._misses = 0
        self._pending: Optional[np.ndarray] = None
        self._pending_count = 0
        self.primary_count = 0
        self.first_primary_ts: Optional[float] = None
        self.last_source = "LOST"
        self.accept_history = collections.deque(maxlen=90)

    @staticmethod
    def _as_array(ellipse: dict) -> np.ndarray:
        return np.asarray([
            ellipse["center"][0], ellipse["center"][1],
            ellipse["axes"][0], ellipse["axes"][1], ellipse["angle"],
        ], dtype=np.float64)

    @staticmethod
    def _as_ellipse(values: np.ndarray) -> dict:
        return {
            "center": (float(values[0]), float(values[1])),
            "axes": (max(1.0, float(values[2])), max(1.0, float(values[3]))),
            "angle": float(values[4]) % 180.0,
        }

    def _prediction(self, timestamp: float) -> Optional[np.ndarray]:
        if self._state is None:
            return None
        predicted = self._state.copy()
        if self._last_measurement_ts is not None:
            dt = min(0.20, max(0.0, timestamp - self._last_measurement_ts))
            predicted[:2] += self._velocity * dt
        return predicted

    @staticmethod
    def _roi_bounds(frame: np.ndarray, roi: Optional[tuple]) -> tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        if roi is None:
            return 0, 0, width, height
        x, y, rw, rh = (int(v) for v in roi)
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        return x, y, max(1, min(width - x, rw)), max(1, min(height - y, rh))

    @staticmethod
    def _ellipse_contrast(frame: np.ndarray, ellipse: dict) -> float:
        """Median surrounding intensity minus median ellipse intensity."""
        cx, cy = ellipse["center"]
        minor_d, major_d = ellipse["axes"]
        radius = int(math.ceil(max(minor_d, major_d) * 0.85)) + 3
        x0 = max(0, int(round(cx)) - radius)
        y0 = max(0, int(round(cy)) - radius)
        x1 = min(frame.shape[1], int(round(cx)) + radius + 1)
        y1 = min(frame.shape[0], int(round(cy)) + radius + 1)
        if x1 - x0 < 5 or y1 - y0 < 5:
            return 0.0
        patch = frame[y0:y1, x0:x1]
        inner = np.zeros(patch.shape, dtype=np.uint8)
        outer = np.zeros(patch.shape, dtype=np.uint8)
        center = (int(round(cx - x0)), int(round(cy - y0)))
        axes_inner = (max(1, int(round(minor_d * 0.42))),
                      max(1, int(round(major_d * 0.42))))
        axes_outer = (max(2, int(round(minor_d * 0.68))),
                      max(2, int(round(major_d * 0.68))))
        angle = float(ellipse["angle"])
        cv2.ellipse(inner, center, axes_inner, angle, 0, 360, 255, -1)
        cv2.ellipse(outer, center, axes_outer, angle, 0, 360, 255, -1)
        ring = cv2.bitwise_and(outer, cv2.bitwise_not(inner))
        inside_values = patch[inner > 0]
        ring_values = patch[ring > 0]
        if inside_values.size < 8 or ring_values.size < 12:
            return 0.0
        return float(np.median(ring_values) - np.median(inside_values))

    def _primary_quality(
        self, result: dict, ellipse: Optional[dict], frame: np.ndarray,
        roi: Optional[tuple], pupil_min: int, pupil_max: int,
        timestamp: float,
    ) -> tuple[bool, str, float]:
        confidence = float(result.get("confidence", 0.0))
        if ellipse is None:
            return False, "no ellipse", 0.0
        if confidence < 0.60:
            return False, f"2D confidence {confidence:.2f}", 0.0
        cx, cy = ellipse["center"]
        minor_d, major_d = ellipse["axes"]
        x, y, rw, rh = self._roi_bounds(frame, roi)
        if not (x <= cx < x + rw and y <= cy < y + rh):
            return False, "outside ROI", 0.0
        if major_d < max(5.0, pupil_min * 0.70) or major_d > pupil_max * 1.15:
            return False, f"diameter {major_d:.0f}px", 0.0
        aspect = minor_d / max(major_d, 1.0)
        if aspect < 0.16:
            return False, f"ellipse aspect {aspect:.2f}", 0.0
        contrast = self._ellipse_contrast(frame, ellipse)
        min_contrast = 5.0 if self._state is not None else 8.0
        if contrast < min_contrast:
            return False, f"dark contrast {contrast:.1f}", contrast

        predicted = self._prediction(timestamp)
        if predicted is not None and self._misses <= self.MAX_STATE_MISSES:
            distance = float(np.linalg.norm(np.asarray([cx, cy]) - predicted[:2]))
            jump_limit = max(32.0, 0.90 * max(major_d, predicted[3]))
            size_ratio = major_d / max(predicted[3], 1.0)
            if distance > jump_limit:
                return False, f"position jump {distance:.0f}px", contrast
            if not 0.42 <= size_ratio <= 2.35:
                return False, f"diameter jump {size_ratio:.2f}x", contrast
        return True, "Detector2D accepted", contrast

    def _confirm_relocation(
        self, result: dict, ellipse: Optional[dict], reason: str, contrast: float,
    ) -> bool:
        """Accept a genuine saccade after two consistent high-quality frames."""
        if ellipse is None or not reason.startswith(("position jump", "diameter jump")):
            self._pending = None
            self._pending_count = 0
            return False
        if float(result.get("confidence", 0.0)) < 0.72 or contrast < 8.0:
            return False
        candidate = self._as_array(ellipse)
        if self._pending is not None:
            tolerance = max(24.0, 0.55 * candidate[3])
            if np.linalg.norm(candidate[:2] - self._pending[:2]) <= tolerance:
                self._pending_count += 1
                self._pending = 0.5 * self._pending + 0.5 * candidate
            else:
                self._pending = candidate
                self._pending_count = 1
        else:
            self._pending = candidate
            self._pending_count = 1
        return self._pending_count >= 2

    def _correct(self, ellipse: dict, timestamp: float, gain: float) -> dict:
        measured = self._as_array(ellipse)
        predicted = self._prediction(timestamp)
        if predicted is None or self._last_measurement_ts is None:
            corrected = measured.copy()
            self._velocity[:] = 0.0
        else:
            dt = max(1.0 / 120.0, min(0.25, timestamp - self._last_measurement_ts))
            raw_velocity = (measured[:2] - self._state[:2]) / dt
            speed = float(np.linalg.norm(raw_velocity))
            if speed > 1600.0:
                raw_velocity *= 1600.0 / speed
            self._velocity = 0.65 * self._velocity + 0.35 * raw_velocity
            corrected = predicted.copy()
            corrected[:4] += gain * (measured[:4] - predicted[:4])
            angle_delta = ((measured[4] - predicted[4] + 90.0) % 180.0) - 90.0
            corrected[4] = (predicted[4] + gain * angle_delta) % 180.0
        self._state = corrected
        self._last_measurement_ts = timestamp
        self._pending = None
        self._pending_count = 0
        return self._as_ellipse(corrected)

    def _find_auxiliary(
        self, frame: np.ndarray, roi: Optional[tuple], pupil_min: int,
        pupil_max: int, timestamp: float,
    ) -> tuple[Optional[dict], float, float]:
        predicted = self._prediction(timestamp)
        if predicted is None:
            return None, 0.0, 0.0
        roi_x, roi_y, roi_w, roi_h = self._roi_bounds(frame, roi)
        search_half = int(min(125, max(58, predicted[3] * 1.65)))
        cx, cy = predicted[:2]
        x0 = max(roi_x, int(round(cx)) - search_half)
        y0 = max(roi_y, int(round(cy)) - search_half)
        x1 = min(roi_x + roi_w, int(round(cx)) + search_half + 1)
        y1 = min(roi_y + roi_h, int(round(cy)) + search_half + 1)
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None, 0.0, 0.0
        patch = cv2.GaussianBlur(frame[y0:y1, x0:x1], (5, 5), 0.9)
        p05, p20, p50 = (float(v) for v in np.percentile(patch, (5, 20, 50)))
        spread = max(8.0, p50 - p05)
        thresholds = sorted({
            int(np.clip(p05 + spread * factor, 2, 245))
            for factor in (0.18, 0.28, 0.38, 0.50)
        })
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        best: Optional[dict] = None
        best_score = 0.0
        best_contrast = 0.0

        for threshold in thresholds:
            mask = cv2.threshold(patch, threshold, 255, cv2.THRESH_BINARY_INV)[1]
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5)
            contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
            for contour in contours:
                if len(contour) < 12:
                    continue
                area = float(cv2.contourArea(contour))
                if area < 35.0:
                    continue
                (local_cx, local_cy), (axis0, axis1), angle = cv2.fitEllipse(contour)
                candidate = _normalise_ellipse({
                    "center": (local_cx + x0, local_cy + y0),
                    "axes": (axis0, axis1),
                    "angle": angle,
                })
                if candidate is None:
                    continue
                minor_d, major_d = candidate["axes"]
                if major_d < max(5.0, pupil_min * 0.65) or major_d > pupil_max * 1.15:
                    continue
                aspect = minor_d / max(major_d, 1.0)
                if aspect < 0.16:
                    continue
                ellipse_area = math.pi * minor_d * major_d * 0.25
                fill = area / max(ellipse_area, 1.0)
                if not 0.30 <= fill <= 1.35:
                    continue
                candidate_center = np.asarray(candidate["center"])
                distance = float(np.linalg.norm(candidate_center - predicted[:2]))
                distance_score = math.exp(-((distance / max(35.0, search_half * 0.72)) ** 2))
                size_score = math.exp(-1.3 * abs(math.log(major_d / max(predicted[3], 1.0))))
                aspect_score = float(np.clip((aspect - 0.12) / 0.55, 0.0, 1.0))
                fill_score = float(np.clip(1.0 - abs(fill - 0.78) / 0.65, 0.0, 1.0))
                contrast = self._ellipse_contrast(frame, candidate)
                contrast_score = float(np.clip((contrast - 4.0) / 28.0, 0.0, 1.0))
                score = (0.31 * contrast_score + 0.28 * distance_score
                         + 0.20 * size_score + 0.13 * aspect_score
                         + 0.08 * fill_score)
                if contrast >= 6.0 and score > best_score:
                    best = candidate
                    best_score = score
                    best_contrast = contrast
        if best_score < 0.60:
            return None, best_score, best_contrast
        return best, best_score, best_contrast

    def process(
        self, result_2d: dict, frame: np.ndarray, roi: Optional[tuple],
        pupil_min: int, pupil_max: int, timestamp: float,
        assist_enabled: bool,
    ) -> TrackingDecision:
        raw_ellipse = _normalise_ellipse(result_2d.get("ellipse"))
        accepted, reason, contrast = self._primary_quality(
            result_2d, raw_ellipse, frame, roi, pupil_min, pupil_max, timestamp
        )
        if not accepted and self._confirm_relocation(result_2d, raw_ellipse, reason, contrast):
            accepted = True
            reason = "confirmed relocation"
            self._state = None
            self._last_measurement_ts = None
        relocation_pending = (
            not accepted and self._pending_count > 0
            and reason.startswith(("position jump", "diameter jump"))
        )

        if accepted and raw_ellipse is not None:
            ellipse = self._correct(raw_ellipse, timestamp, gain=0.58)
            self._misses = 0
            self.primary_count += 1
            if self.first_primary_ts is None:
                self.first_primary_ts = timestamp
            self.last_source = "PRIMARY"
            self.accept_history.append(True)
            return TrackingDecision(
                ellipse, "PRIMARY", float(result_2d.get("confidence", 0.0)),
                True, reason, contrast,
            )

        self.accept_history.append(False)
        # During a possible true saccade, hold one predicted frame while the
        # second Detector2D frame confirms the new location. Searching near the
        # old location here can otherwise lock the auxiliary detector to iris.
        if assist_enabled and self._state is not None and not relocation_pending:
            auxiliary, score, aux_contrast = self._find_auxiliary(
                frame, roi, pupil_min, pupil_max, timestamp
            )
            if auxiliary is not None:
                ellipse = self._correct(auxiliary, timestamp, gain=0.42)
                self._misses = min(self.MAX_STATE_MISSES, self._misses + 1)
                self.last_source = "ASSIST"
                return TrackingDecision(
                    ellipse, "ASSIST", min(0.79, 0.70 + score * 0.09),
                    False, reason, aux_contrast,
                )

        self._misses += 1
        predicted = self._prediction(timestamp)
        if predicted is not None and self._misses <= self.MAX_PREDICT_FRAMES:
            self.last_source = "PREDICT"
            return TrackingDecision(
                self._as_ellipse(predicted), "PREDICT", 0.0,
                False, reason, contrast,
            )
        if self._misses > self.MAX_STATE_MISSES:
            self._state = None
            self._velocity[:] = 0.0
            self._last_measurement_ts = None
        self.last_source = "LOST"
        return TrackingDecision(None, "LOST", 0.0, False, reason, contrast)

    @property
    def accepted_ratio(self) -> float:
        if not self.accept_history:
            return 0.0
        return float(sum(self.accept_history)) / len(self.accept_history)


class TemporalPupilAssist(_TemporalPupilBase):
    """Visibility gate for oblique eye images; no gaze through missing pupils.

    Missing evidence is labelled OCCLUDED/UNCERTAIN, not a confirmed blink.
    Auxiliary ellipses are display-only. Three consecutive supported primary
    measurements are required after loss, with no stale velocity on recovery.
    """

    def reset(self):
        super().reset()
        self.visibility = "SEARCHING"
        self.edge_support = 0.0
        self._good_run = 0
        self._candidate = None
        self._last_primary_ts = None
        self._process_ts = None
        self._reference_areas = collections.deque(maxlen=45)
        self.gaze_allowed = False
        self._local_detector = None
        self._retry_frame = 0

    def refine_primary(self, result, frame, roi, pupil_min, pupil_max):
        """Retry native Detector2D within an evidence-based local ROI.

        All returned ellipses remain real Detector2D observations in full-frame
        coordinates. Dark auxiliary contours are never used to train pye3d.
        """
        initial = _normalise_ellipse(result.get("ellipse"))
        initial_support = self.boundary_support(frame, initial)
        if float(result.get("confidence", 0)) >= .85 and initial_support >= .65:
            return result
        seeds = []
        if initial is not None and initial_support >= .40:
            seeds.append(initial)
        if self._state is not None:
            seeds.append(self._as_ellipse(self._state))
        self._retry_frame += 1
        if self._retry_frame % 3 == 0:
            seeds.extend(self._image_seeds(frame, roi, pupil_min, pupil_max))
        if self._local_detector is None:
            self._local_detector = Detector2D()
        best = result
        best_score = float(result.get("confidence", 0)) * initial_support
        x, y, w, h = self._roi_bounds(frame, roi)
        # A failed raw detection used to disable recovery entirely. Periodically
        # rescan the user's complete ROI using an independent native detector.
        # Keep native contour observations intact for pye3d; never synthesize them.
        if self._retry_frame % 3 == 0:
            for intensity in (15, 35):
                self._local_detector.update_properties(dict(
                    coarse_detection=True, coarse_filter_min=100, coarse_filter_max=8000,
                    pupil_size_min=pupil_min, pupil_size_max=pupil_max,
                    intensity_range=intensity))
                candidate = self._local_detector.detect(frame, roi=Roi(x, y, x+w, y+h))
                support = self.boundary_support(frame, candidate.get("ellipse"))
                score = float(candidate.get("confidence", 0)) * support
                if support >= .65 and score > best_score:
                    best, best_score = candidate, score
        for seed in seeds[:3]:
            cx, cy = seed["center"]
            margin = max(45, min(110, max(seed["axes"]) * 1.1))
            left, top = max(x, int(cx-margin)), max(y, int(cy-margin))
            right, bottom = min(x+w, int(cx+margin)), min(y+h, int(cy+margin))
            if right-left < 12 or bottom-top < 12:
                continue
            self._local_detector.update_properties(dict(
                coarse_detection=True, coarse_filter_min=100, coarse_filter_max=8000,
                pupil_size_min=max(5, int(pupil_min * .5)), pupil_size_max=pupil_max, intensity_range=23))
            candidate = self._local_detector.detect(frame, roi=Roi(left, top, right, bottom))
            support = self.boundary_support(frame, candidate.get("ellipse"))
            score = float(candidate.get("confidence", 0)) * support
            if support >= .60 and score > best_score:
                best, best_score = candidate, score
        return best

    def _image_seeds(self, frame, roi, pupil_min, pupil_max):
        """Dark connected components guide native detection, never become gaze.

        An enclosing iris boundary is necessary; skin shadows touching the
        search boundary and thin eyelashes are excluded. No prior lock needed.
        """
        x, y, w, h = self._roi_bounds(frame, roi)
        patch = cv2.GaussianBlur(frame[y:y+h, x:x+w], (5, 5), 0)
        ranked = []
        for threshold in (25, 50, 80):
            mask = cv2.threshold(patch, threshold, 255, cv2.THRESH_BINARY_INV)[1]
            for contour in cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]:
                if len(contour) < 12 or cv2.contourArea(contour) < 40:
                    continue
                bx, by, bw, bh = cv2.boundingRect(contour)
                if bx <= 1 or by <= 1 or bx+bw >= w-1 or by+bh >= h-1:
                    continue
                center, axes, angle = cv2.fitEllipse(contour)
                el = _normalise_ellipse(dict(center=(center[0]+x, center[1]+y), axes=axes, angle=angle))
                if el is None or not max(8, pupil_min*.7) <= max(axes) <= pupil_max:
                    continue
                if min(axes)/max(axes) < .16:
                    continue
                fill = cv2.contourArea(contour) / max(1, math.pi*np.prod(axes)/4)
                support = self.boundary_support(frame, el)
                if .65 <= fill <= 1.2 and support >= .65 and self._ellipse_contrast(frame, el) >= 12:
                    ranked.append((support, el))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        unique = []
        for _, el in ranked:
            if all(np.linalg.norm(np.array(el["center"])-np.array(other["center"])) > 15 for other in unique):
                unique.append(el)
            if len(unique) >= 2:
                break
        return unique

    @staticmethod
    def boundary_support(frame, ellipse):
        """Fraction of angular sectors with a dark-to-light pupil boundary.

        Samples use the ellipse's own axes, so a narrow oblique pupil is not
        forced into a circular template. Requires evidence in all quadrants.
        """
        el = _normalise_ellipse(ellipse)
        if el is None:
            return 0.0
        angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        a, b = np.asarray(el["axes"], dtype=float) / 2
        rotation = math.radians(el["angle"])
        dx = a * np.cos(angles) * math.cos(rotation) - b * np.sin(angles) * math.sin(rotation)
        dy = a * np.cos(angles) * math.sin(rotation) + b * np.sin(angles) * math.cos(rotation)
        values, valid = [], np.ones(64, dtype=bool)
        for scale in (0.72, 1.22):
            x = el["center"][0] + scale * dx
            y = el["center"][1] + scale * dy
            valid &= (x >= 1) & (x < frame.shape[1] - 1) & (y >= 1) & (y < frame.shape[0] - 1)
            values.append(cv2.remap(frame, x.astype(np.float32)[None, :],
                                     y.astype(np.float32)[None, :], cv2.INTER_LINEAR)[0].astype(float))
        supported = valid & ((values[1] - values[0]) >= 8.0)
        quadrants = supported.reshape(4, 16).mean(axis=1)
        return float(min(supported.mean(), quadrants.min() + 0.25))

    def process(self, result_2d, frame, roi, pupil_min, pupil_max, timestamp, assist_enabled):
        self.gaze_allowed = False
        if not math.isfinite(timestamp):
            self.visibility = "UNCERTAIN"
            return TrackingDecision(None, "LOST", 0.0, False, "invalid timestamp")
        if self._process_ts is not None and (timestamp <= self._process_ts or timestamp - self._process_ts > 0.25):
            self._good_run = 0
            self._candidate = None
            self._velocity[:] = 0
        self._process_ts = timestamp
        ellipse = _normalise_ellipse(result_2d.get("ellipse"))
        valid, reason, contrast = self._primary_quality(
            result_2d, ellipse, frame, roi, pupil_min, pupil_max, timestamp)
        confidence = float(result_2d.get("confidence", 0.0))
        self.edge_support = self.boundary_support(frame, ellipse)
        # Continuity alone never validates a pupil. A slightly lower native
        # score is allowed only with stronger current image-boundary evidence.
        minimum_confidence = .60 if self.edge_support >= .80 and contrast >= 12 else .70
        if not math.isfinite(confidence) or confidence < minimum_confidence:
            valid, reason = False, "weak primary evidence"
        elif self.edge_support < 0.60:
            valid, reason = False, "pupil boundary partly hidden"
        if ellipse is not None and self._reference_areas:
            area = float(np.prod(ellipse["axes"]))
            reference = float(np.median(self._reference_areas))
            # Perspective can greatly reduce ellipse area during an eye turn.
            # Reject collapse with weak boundaries, not a well-supported ellipse.
            if area < reference * 0.45 and self.edge_support < .80:
                valid, reason = False, "abrupt pupil-area collapse"
        # A new gaze direction can relocate, but needs repeated measurements.
        if (not valid and reason.startswith(("position jump", "diameter jump"))
                and confidence >= 0.72 and self.edge_support >= 0.65 and contrast >= 8):
            valid = True
            self._good_run = 0 if self._candidate is None else self._good_run
        if valid and ellipse is not None:
            center = np.asarray(ellipse["center"])
            if self._candidate is not None and np.linalg.norm(center - self._candidate) > max(24, max(ellipse["axes"]) * 0.65):
                self._good_run = 0
            self._candidate = center
            self._good_run += 1
            if self._good_run >= 3:
                if self.visibility != "VISIBLE":
                    self._state = None
                    self._last_measurement_ts = None
                filtered = self._correct(ellipse, timestamp, gain=0.70)
                self._last_primary_ts = timestamp
                self._misses = 0
                self.primary_count += 1
                if self.first_primary_ts is None:
                    self.first_primary_ts = timestamp
                self._reference_areas.append(float(np.prod(ellipse["axes"])))
                self.visibility, self.last_source = "VISIBLE", "PRIMARY"
                self.gaze_allowed = True
                self.accept_history.append(True)
                return TrackingDecision(filtered, "PRIMARY", confidence, True, "visible pupil", contrast)
            self.visibility, self.last_source = "RECOVERING", "LOST"
            self.accept_history.append(False)
            return TrackingDecision(ellipse, "LOST", 0.0, False, "confirming visible pupil", contrast)

        self._good_run = 0
        self._candidate = None
        self._velocity[:] = 0
        self._misses += 1
        self.accept_history.append(False)
        self.visibility = "OCCLUDED/UNCERTAIN"
        elapsed = float("inf") if self._last_primary_ts is None else timestamp - self._last_primary_ts
        if assist_enabled and self._state is not None and elapsed <= 0.25:
            auxiliary, score, aux_contrast = self._find_auxiliary(frame, roi, pupil_min, pupil_max, timestamp)
            if auxiliary is not None and self.boundary_support(frame, auxiliary) >= 0.65:
                # Do not correct temporal state or feed pye3d from this candidate.
                self.last_source = "ASSIST"
                return TrackingDecision(auxiliary, "ASSIST", min(score, 0.59), False,
                                        "display-only auxiliary; gaze paused", aux_contrast)
        if elapsed > 0.6:
            self._state = None
            self._last_measurement_ts = None
            self._reference_areas.clear()
        self.last_source = "LOST"
        return TrackingDecision(None, "LOST", 0.0, False, reason, contrast)


# ── camera discovery ───────────────────────────────────────────────────────────

# def _find_pupil_uids() -> list[str]:
#     """
#     Return UIDs of Pupil eye cameras in bus order.
#     Two cameras with the same product name are distinguished by bus position.
#     """
    
#     devs = uvc.device_list()
#     pupil = [d for d in devs if "Pupil" in d.get("name", "")]
#     # sort by (bus_number, device_address) for stable eye 0 / eye 1 assignment
#     pupil.sort(key=lambda d: (d.get("bus_number", 0), d.get("device_address", 0)))
#     return [d["uid"] for d in pupil]

def _find_pupil_uids() -> list[str]:
    """Return only this headset's two Cam2 sensors in stable USB bus order."""
    pupil = [
        device
        for device in uvc.device_list()
        if device.get("idVendor") == 0x0C45
        and device.get("idProduct") == 0x64AB
    ]
    pupil.sort(key=lambda device: (
        int(device.get("bus_number", 0)),
        int(device.get("device_address", 0)),
        str(device.get("uid", "")),
    ))
    return [str(device["uid"]) for device in pupil]

def _pick_mode(cap: uvc.Capture, want: Optional[str] = None) -> Optional[uvc.CaptureMode]:
    """Pick a capture mode.

    If *want* is a label string (from _mode_label), find that exact mode.
    Otherwise prefer the headset-native 400x400 @ 30 fps MJPG mode.
    """
    modes = cap.available_modes
    if not modes:
        return None
    if want is not None:
        for m in modes:
            if _mode_label(m) == want:
                return m
    for fmt in PREFERRED_FORMATS:
        for m in modes:
            if (m.width, m.height, m.fps) == (TARGET_W, TARGET_H, TARGET_FPS) \
                    and m.format_name == fmt:
                return m
    for fmt in PREFERRED_FORMATS:
        candidates = [m for m in modes if m.format_name == fmt]
        if candidates:
            return min(candidates,
                       key=lambda m: (abs(m.width - TARGET_W)
                                      + abs(m.height - TARGET_H),
                                      abs(m.fps - TARGET_FPS)))
    return min(modes, key=lambda m: (abs(m.width - TARGET_W)
                                     + abs(m.height - TARGET_H),
                                     abs(m.fps - TARGET_FPS)))


def _mode_label(m: uvc.CaptureMode) -> str:
    """Human-readable label for a capture mode, e.g. '192×192  30fps  GREY'."""
    return f"{m.width}\u00d7{m.height}  {m.fps}fps  {m.format_name}"


# ── per-eye capture thread ─────────────────────────────────────────────────────

class EyeCapture(threading.Thread):
    def __init__(self, eye_id: int, uid: str,
                 want_mode: Optional[str] = None,
                 on_modes_ready: Optional[callable] = None,
                 on_exposure_ready: Optional[callable] = None) -> None:
        super().__init__(daemon=True, name=f"eye-{eye_id}")
        self.eye_id           = eye_id
        self.uid              = uid
        self.want_mode        = want_mode
        self.on_modes_ready   = on_modes_ready
        self.on_exposure_ready = on_exposure_ready
        self._lock    = threading.Lock()
        self._frame:  Optional[np.ndarray] = None
        self._seq     = 0
        self._timestamp = 0.0
        self._fps     = 0.0
        self._running = False
        self.error:   str = ""
        self.width    = 0
        self.height   = 0
        self.available_mode_labels: list[str] = []
        self.active_mode_label: str = ""
        self._exp_auto_ctrl = None
        self._exp_val_ctrl  = None
        self._control_commands: queue.Queue[tuple[bool, int]] = queue.Queue()
        self.exp_min = 0
        self.exp_max = 500

    # ── public ────────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._running = False

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_latest(self) -> tuple[Optional[np.ndarray], int, float]:
        """Atomically copy the newest image, sequence number, and capture time."""
        with self._lock:
            image = self._frame.copy() if self._frame is not None else None
            return image, self._seq, self._timestamp

    def get_seq(self) -> int:
        return self._seq

    def get_fps(self) -> float:
        return self._fps

    def apply_exposure(self, auto: bool, value: int) -> None:
        """Queue exposure settings for application by the capture thread.

        Auto Exposure Mode UVC enum: 1=Manual, 2=Auto, 4=Shutter Priority, 8=Aperture Priority.
        """
        self._control_commands.put((bool(auto), int(value)))

    def _apply_pending_exposure(self) -> None:
        """Apply only the newest queued exposure command inside this thread."""
        latest = None
        while True:
            try:
                latest = self._control_commands.get_nowait()
            except queue.Empty:
                break
        if latest is None:
            return
        auto, value = latest
        if self._exp_auto_ctrl is not None:
            try:
                target_mode = 2 if auto else 1
                if int(self._exp_auto_ctrl.value) != target_mode:
                    self._exp_auto_ctrl.value = target_mode
            except Exception:
                pass
        if not auto and self._exp_val_ctrl is not None:
            try:
                self._exp_val_ctrl.value = max(self.exp_min,
                                               min(self.exp_max, value))
            except Exception:
                pass

    # ── capture loop ──────────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True
        try:
            cap = uvc.Capture(self.uid)
        except Exception as exc:
            self.error = f"eye{self.eye_id}: open failed – {exc}"
            return

        # publish available modes to the UI
        self.available_mode_labels = [_mode_label(m) for m in cap.available_modes]
        if self.on_modes_ready:
            self.on_modes_ready(self.available_mode_labels)

        mode = _pick_mode(cap, self.want_mode)
        if mode is None:
            self.error = f"eye{self.eye_id}: no capture modes available"
            cap.close()
            return

        try:
            cap.frame_mode = mode
        except Exception as exc:
            self.error = f"eye{self.eye_id}: set mode failed – {exc}"
            cap.close()
            return

        self.width  = cap.frame_mode.width
        self.height = cap.frame_mode.height
        self.active_mode_label = _mode_label(cap.frame_mode)

        # discover exposure controls and report ranges to the UI
        for ctrl in cap.controls:
            name = ctrl.display_name.lower()
            if name == "auto exposure mode":
                self._exp_auto_ctrl = ctrl
            elif name == "absolute exposure time":
                self._exp_val_ctrl = ctrl
        if self._exp_val_ctrl is not None:
            self.exp_min    = max(0, int(self._exp_val_ctrl.min_val))
            self.exp_max    = max(self.exp_min + 1, int(self._exp_val_ctrl.max_val))
            current_exp     = max(self.exp_min, int(self._exp_val_ctrl.value))
        else:
            current_exp = 0
        if self.on_exposure_ready:
            self.on_exposure_ready(self.exp_min, self.exp_max, current_exp)

        fps_t = time.monotonic()
        fps_n = 0

        while self._running:
            self._apply_pending_exposure()
            try:
                frame = cap.get_frame(timeout=0.5)
            except Exception as e:
                print(f"eye{self.eye_id} get_frame error:", repr(e))
                time.sleep(0.05)
                continue
            

            if frame is None:
                continue

            gray = frame.gray
            if gray is None:
                continue

            capture_timestamp = time.monotonic()
            with self._lock:
                # libuvc can reuse the frame buffer after get_frame returns.
                self._frame = np.ascontiguousarray(gray).copy()
                self._seq  += 1
                self._timestamp = capture_timestamp

            fps_n += 1
            now = time.monotonic()
            if now - fps_t >= 1.0:
                self._fps = fps_n / (now - fps_t)
                fps_n = 0
                fps_t = now

        cap.close()


# ── gaze top-down view ────────────────────────────────────────────────────────

class GazeTopDownView:
    """
    Separate Toplevel window – top-down (bird's-eye) view of both eyes and
    their gaze rays projected onto a horizontal plane.

    Owns:
      IPD slider, mount-dist slider, screen-depth spinbox, cam/toe-in angle sliders.
    Receives:
      update(idx, theta, phi, conf) called from the main render loop.
    """

    CANVAS_W = 440
    CANVAS_H = 520

    def __init__(self, container: ttk.Frame,
                 ipd_var: tk.IntVar,
                 mount_var: tk.IntVar) -> None:
        self.frame = ttk.LabelFrame(container, text="Gaze \u2013 Top-Down View", padding=4)
        self._alive   = True
        self._visible = False
        self.ipd_var   = ipd_var
        self.mount_var = mount_var
        self._ipd_str   = tk.StringVar(value=str(ipd_var.get()))
        self._mnt_str   = tk.StringVar(value=str(mount_var.get()))
        ipd_var.trace_add("write",
            lambda *_: self._ipd_str.set(str(self.ipd_var.get())))
        mount_var.trace_add("write",
            lambda *_: self._mnt_str.set(str(self.mount_var.get())))
        self.cam_angle_var    = tk.IntVar(value=68)   # camera-to-eye-axis angle (°)
        self.screen_tilt_var  = tk.IntVar(value=10)   # eyepiece toe-in (°)
        self.screen_depth_var = tk.IntVar(value=20)   # eye-to-screen distance (pancake optics)
        self._cam_str    = tk.StringVar(value="68")
        self._tilt_str   = tk.StringVar(value="10")
        self._depth_str  = tk.StringVar(value="20")
        self.cam_angle_var.trace_add("write",
            lambda *_: self._cam_str.set(str(self.cam_angle_var.get())))
        self.screen_tilt_var.trace_add("write",
            lambda *_: self._tilt_str.set(str(self.screen_tilt_var.get())))
        self.screen_depth_var.trace_add("write",
            lambda *_: self._depth_str.set(str(self.screen_depth_var.get())))
        self._gaze: list[Optional[tuple]] = [None, None]  # (theta, phi, conf)
        self._gaze_updated = [0.0, 0.0]
        self._build_ui()
        self._tick()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.canvas = tk.Canvas(
            self.frame, width=self.CANVAS_W, height=self.CANVAS_H,
            bg="#0d1117", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 2))
        ttk.Label(self.frame, text="UNCALIBRATED geometry preview - not a screen measurement",
                  foreground="#e0aa55", wraplength=420).pack(padx=4)

        ctrl = ttk.Frame(self.frame, padding=(4, 4, 4, 4))
        ctrl.pack(fill=tk.X)
        ctrl.columnconfigure(1, weight=1)

        ttk.Label(ctrl, text="IPD (mm):").grid(row=0, column=0, sticky=tk.W)
        ttk.Scale(ctrl, variable=self.ipd_var, from_=50, to=80,
                  orient=tk.HORIZONTAL, length=180).grid(
                  row=0, column=1, padx=(6, 4), sticky=tk.EW)
        ttk.Label(ctrl, textvariable=self._ipd_str, width=4,
                  anchor=tk.E).grid(row=0, column=2)

        ttk.Label(ctrl, text="Mount dist (mm):").grid(
                  row=1, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Scale(ctrl, variable=self.mount_var, from_=50, to=2000,
                  orient=tk.HORIZONTAL, length=180).grid(
                  row=1, column=1, padx=(6, 4), pady=(6, 0), sticky=tk.EW)
        ttk.Label(ctrl, textvariable=self._mnt_str, width=5,
                  anchor=tk.E).grid(row=1, column=2, pady=(6, 0))

        # Camera + screen geometry (hardware-specific angles)
        geom = ttk.Frame(ctrl)
        geom.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=(6, 0))
        ttk.Label(geom, text="Cam angle (°):").pack(side=tk.LEFT)
        ttk.Scale(geom, variable=self.cam_angle_var, from_=0, to=90,
                  orient=tk.HORIZONTAL, length=100).pack(side=tk.LEFT, padx=(4, 2))
        ttk.Label(geom, textvariable=self._cam_str, width=3).pack(side=tk.LEFT)
        ttk.Label(geom, text="  Toe-in (°):").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Scale(geom, variable=self.screen_tilt_var, from_=0, to=30,
                  orient=tk.HORIZONTAL, length=80).pack(side=tk.LEFT, padx=(4, 2))
        ttk.Label(geom, textvariable=self._tilt_str, width=3).pack(side=tk.LEFT)

        geom2 = ttk.Frame(ctrl)
        geom2.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))
        ttk.Label(geom2, text="Screen depth (mm, pancake):").pack(side=tk.LEFT)
        ttk.Spinbox(geom2, textvariable=self.screen_depth_var,
                    from_=5, to=80, width=4).pack(side=tk.LEFT, padx=(6, 0))

    # ── data ──────────────────────────────────────────────────────────────────

    def update(self, idx: int, theta: float, phi: float, conf: float) -> None:
        """Called from main render loop with the latest pye3d result."""
        self._gaze[idx] = (theta, phi, conf)
        self._gaze_updated[idx] = time.monotonic()

    def invalidate(self, idx: int) -> None:
        """Mark eye idx gaze as unavailable (detection off or low confidence)."""
        self._gaze[idx] = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def show(self) -> None:
        self._visible = True
        self.frame.grid(row=0, column=2, padx=(10, 0), sticky="ns")

    def hide(self) -> None:
        self._visible = False
        self.frame.grid_remove()

    def _tick(self) -> None:
        if not self._alive:
            return
        try:
            self._draw()
        except Exception:
            pass
        self.frame.after(33, self._tick)

    # ── drawing ───────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        c = self.canvas
        c.delete("all")
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 20 or H < 20:
            return

        ipd_mm    = float(max(40,  self.ipd_var.get()))
        mount_mm  = float(max(10,  self.mount_var.get()))
        scr_d     = float(max(5,   self.screen_depth_var.get()))
        cam_rad   = math.radians(float(self.cam_angle_var.get()))
        toe_rad   = math.radians(float(self.screen_tilt_var.get()))
        cos_cam, sin_cam = math.cos(cam_rad), math.sin(cam_rad)
        cos_toe, sin_toe = math.cos(toe_rad), math.sin(toe_rad)

        # World: origin = midpoint between eyes, +X rightward, +Y forward.
        # Canvas: ox/oy = world origin; world +Y → canvas up (decreasing cy).
        ppm = (H * 0.60) / mount_mm
        ox  = W / 2.0
        oy  = H * 0.88

        def w2c(xmm: float, ymm: float):
            return ox + xmm * ppm, oy - ymm * ppm

        # ── depth guide lines ─────────────────────────────────────────────────
        for frac in (0.25, 0.5, 0.75, 1.0, 1.5):
            d = mount_mm * frac
            _, cy = w2c(0, d)
            if -10 < cy < H + 10:
                c.create_line(0, cy, W, cy, fill="#111a2a", width=1)
                c.create_text(6, cy - 7, text=f"{d:.0f}",
                              fill="#1e3044", anchor=tk.W, font=("Consolas", 7))

        # ── mount reference line ──────────────────────────────────────────────
        _, mount_cy = w2c(0, mount_mm)
        c.create_line(0, mount_cy, W, mount_cy, fill="#1a3a1a", width=2, dash=(8, 4))
        c.create_text(W - 6, mount_cy - 10,
                      text=f"mount  {mount_mm:.0f} mm",
                      fill="#2a8c2a", anchor=tk.NE, font=("Consolas", 8))

        # ── head outline ──────────────────────────────────────────────────────
        hw  = ipd_mm * 0.85 * ppm
        hbk = ipd_mm * 1.10 * ppm
        hfw = ipd_mm * 0.60 * ppm
        c.create_oval(ox - hw, oy - hfw, ox + hw, oy + hbk,
                      outline="#222244", width=2)
        nx, ny = w2c(0, ipd_mm * 0.52)
        c.create_polygon(ox - 6, ny + 10, ox + 6, ny + 10, ox, ny - 3,
                         fill="#222244", outline="")

        COLORS = ["#3399ff", "#ff9933"]
        EYE_R  = max(6, int(ipd_mm * ppm * 0.10))
        world_rays = []   # (eye_wx, gx_h, gz_h)  for convergence calc

        for i in range(2):
            sign   = -1 if i == 0 else 1    # eye 0 = Left, eye 1 = Right
            eye_wx = sign * ipd_mm / 2
            ecx, ecy = w2c(eye_wx, 0)
            col    = COLORS[i]
            label  = "L" if i == 0 else "R"

            # eyeball circle
            c.create_oval(ecx - EYE_R, ecy - EYE_R,
                          ecx + EYE_R, ecy + EYE_R,
                          outline=col, width=2)
            c.create_text(ecx, ecy + EYE_R + 10,
                          text=f"Eye{i} ({label})", fill=col, font=("Consolas", 7))

            # eyepiece axis (unit vector): toed inward by toe_rad
            ep_x = -sign * sin_toe   # +rightward component
            ep_y =  cos_toe          # +forward component

            # screen marker: segment at scr_d mm along eyepiece axis
            sc_wx = eye_wx + scr_d * ep_x
            sc_wy = scr_d * ep_y
            sc_cx, sc_cy = w2c(sc_wx, sc_wy)
            # perpendicular to ep_axis (rotated 90 deg CW in world)
            perp_wx, perp_wy = ep_y, -ep_x
            sc_half_px = 10
            perp_canvas_x =  perp_wx * sc_half_px
            perp_canvas_y = -perp_wy * sc_half_px   # world +y → canvas up
            c.create_line(sc_cx - perp_canvas_x, sc_cy - perp_canvas_y,
                          sc_cx + perp_canvas_x, sc_cy + perp_canvas_y,
                          fill=col, width=3)
            c.create_text(sc_cx + sign * (sc_half_px + 3), sc_cy,
                          text="scr", fill=col, font=("Consolas", 6),
                          anchor=tk.E if i == 0 else tk.W)

            gaze = self._gaze[i]
            if time.monotonic() - self._gaze_updated[i] > 0.25:
                gaze = None
            if gaze is None or gaze[2] < 0.4:
                # default ray: this eye pointing straight at screen center
                dx = -eye_wx          # toward X=0
                dy = mount_mm         # toward mount distance
                mag = math.sqrt(dx * dx + dy * dy) + 1e-9
                gx_def, gz_def = dx / mag, dy / mag
                world_rays.append((eye_wx, gx_def, gz_def, False))
                # Missing eyes have no gaze ray, including no default direction.
                c.create_text(ecx, ecy, text="?",
                              fill=col, font=("Consolas", 10, "bold"))
                continue

            theta, phi, conf = gaze

            # Camera-frame horizontal gaze components
            gx_c, _, gz_c = _pye3d_direction(theta, phi)

            # Rotate camera frame -> head frame  R_y(sign * cam_angle)
            # Right eye (sign=+1): camera is on nasal (left) side of eye
            # Left  eye (sign=-1): camera is on nasal (right) side of eye
            gx_h = cos_cam * gx_c + sign * sin_cam * gz_c
            gz_h = -sign * sin_cam * gx_c + cos_cam * gz_c
            mag  = math.sqrt(gx_h ** 2 + gz_h ** 2) + 1e-9
            gx_h /= mag
            gz_h /= mag

            world_rays.append((eye_wx, gx_h, gz_h, True))

            # pupil dot inside eyeball
            shift = EYE_R * 0.55
            c.create_oval(ecx + gx_h * shift - 4, ecy - gz_h * shift - 4,
                          ecx + gx_h * shift + 4, ecy - gz_h * shift + 4,
                          fill=col, outline="")

            # gaze ray (dashed, to canvas edge)
            far = max(W, H) * 3
            c.create_line(ecx, ecy,
                          ecx + gx_h * far * ppm / ppm,   # keep as-is: just scale
                          ecy - gz_h * far * ppm / ppm,
                          fill=col, width=2, dash=(6, 3))
            # arrow head at canvas clip
            t_edge = max(W, H) * 2
            tip_x = ecx + gx_h * t_edge
            tip_y = ecy - gz_h * t_edge
            if 0 <= tip_x <= W and 0 <= tip_y <= H:
                c.create_line(ecx, ecy, tip_x, tip_y,
                              fill=col, width=2, arrow=tk.LAST, arrowshape=(10, 12, 4))

        # ── convergence point (always shown; default rays fill in missing eyes) ─
        # world_rays always has 2 entries: (eye_wx, gx_h, gz_h, measured)
        if len(world_rays) == 2:
            (x1, gx1, gz1, m1) = world_rays[0]
            (x2, gx2, gz2, m2) = world_rays[1]
            measured_count = int(m1) + int(m2)
            # style by reliability: both=white, one=dim, neither=very dim
            if measured_count == 2:
                pt_col, lbl_col, pt_w = "white", "white", 2
            elif measured_count == 1:
                pt_col, lbl_col, pt_w = "#888888", "#888888", 1
            else:
                pt_col, lbl_col, pt_w = "#334455", "#445566", 1

            denom = gx1 * gz2 - gz1 * gx2
            if abs(denom) > 1e-6:
                t1 = (x2 - x1) * gz2 / denom
                t2 = (x2 - x1) * gz1 / denom
                if t1 > 0 and t2 > 0:
                    conv_wx = x1 + t1 * gx1
                    conv_wy = t1 * gz1
                    conv_cx, conv_cy = w2c(conv_wx, conv_wy)

                    dot = max(-1.0, min(1.0, gx1 * gx2 + gz1 * gz2))
                    verg_deg = math.degrees(math.acos(dot))

                    R = 10
                    c.create_oval(conv_cx - R, conv_cy - R,
                                  conv_cx + R, conv_cy + R,
                                  outline=pt_col, width=pt_w)
                    c.create_line(conv_cx - R - 5, conv_cy,
                                  conv_cx + R + 5, conv_cy, fill=pt_col, width=pt_w)
                    c.create_line(conv_cx, conv_cy - R - 5,
                                  conv_cx, conv_cy + R + 5, fill=pt_col, width=pt_w)

                    src_tag = ("both eyes" if measured_count == 2
                               else "1 eye + default" if measured_count == 1
                               else "default")
                    lbl = (f"X={conv_wx:+.1f} mm\n"
                           f"Y={conv_wy:.1f} mm\n"
                           f"vergence {verg_deg:.1f}°  [{src_tag}]")
                    anchor = tk.SW if conv_cx < W * 0.6 else tk.SE
                    ox_lbl = conv_cx + 14 if anchor == tk.SW else conv_cx - 14
                    c.create_text(ox_lbl, conv_cy - 14,
                                  text=lbl, fill=lbl_col,
                                  anchor=anchor, font=("Consolas", 8, "bold"))
                else:
                    c.create_text(W / 2, 20, text="gaze rays diverge",
                                  fill="#885522", font=("Consolas", 8),
                                  anchor=tk.N)
            else:
                c.create_text(W / 2, 20, text="∞  parallel gaze",
                              fill="#448844", font=("Consolas", 9), anchor=tk.N)

        c.create_text(W / 2, H - 7, text="─── Top-Down View ───",
                      fill="#1a2a3a", font=("Consolas", 8))


class DetectionDebugView:
    """Pupil-Capture-style algorithm inspection window for both eye cameras."""

    MODES = (
        "Algorithm Overlay",
        "Dark-Pixel Mask",
        "Edges",
        "Raw IR",
    )

    def __init__(self, root: tk.Tk, app: "PupilDualStreamApp") -> None:
        self.app = app
        self.window = tk.Toplevel(root)
        self.window.title("Pupil 2D/3D Detection Debug")
        self.window.protocol("WM_DELETE_WINDOW", self.hide)
        self.window.resizable(True, True)
        self._visible = False
        self._packets: list[Optional[dict]] = [None, None]
        self._photos: list[Optional[ImageTk.PhotoImage]] = [None, None]
        self.mode_var = tk.StringVar(value=self.MODES[0])
        self.status_var = tk.StringVar(
            value="Open the view for a few seconds before Auto Tune."
        )
        self._canvases: list[tk.Canvas] = []
        self._info_vars: list[tk.StringVar] = []
        self._build_ui()
        self.window.withdraw()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.window, padding=(10, 8, 10, 6))
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, text="View:").pack(side=tk.LEFT)
        mode_box = ttk.Combobox(
            toolbar,
            textvariable=self.mode_var,
            values=self.MODES,
            state="readonly",
            width=20,
        )
        mode_box.pack(side=tk.LEFT, padx=(6, 12))
        mode_box.bind("<<ComboboxSelected>>", lambda _event: self.redraw())
        ttk.Button(
            toolbar,
            text="Auto Tune 2D + Apply",
            command=self.app._auto_tune_detection,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            toolbar,
            text="Suggest ROI Eye 0",
            command=lambda: self.app._apply_suggested_roi(0),
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            toolbar,
            text="Suggest ROI Eye 1",
            command=lambda: self.app._apply_suggested_roi(1),
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            toolbar,
            text="Save Debug Pack",
            command=self.app._save_debug_pack,
        ).pack(side=tk.LEFT)

        body = ttk.Frame(self.window, padding=(10, 4, 10, 4))
        body.pack(fill=tk.BOTH, expand=True)
        for index in range(2):
            frame = ttk.LabelFrame(body, text=f"Eye {index}", padding=6)
            frame.grid(row=0, column=index, padx=(0, 8 if index == 0 else 0))
            canvas = tk.Canvas(
                frame,
                width=400,
                height=400,
                bg="#0d1117",
                highlightthickness=0,
            )
            canvas.pack()
            self._canvases.append(canvas)
            info = tk.StringVar(value="Waiting for debug data…")
            ttk.Label(
                frame,
                textvariable=info,
                font=("Consolas", 8),
                justify=tk.LEFT,
            ).pack(fill=tk.X, anchor=tk.W, pady=(5, 0))
            self._info_vars.append(info)

        ttk.Label(
            self.window,
            textvariable=self.status_var,
            font=("Segoe UI", 9, "bold"),
            padding=(10, 4, 10, 4),
        ).pack(fill=tk.X)
        ttk.Label(
            self.window,
            text=(
                "Recommended setup: make the ROI as small as possible while retaining "
                "the pupil through the full gaze range. In Dark-Pixel Mask, blue should "
                "cover the pupil completely with minimal leakage onto lashes, shadows, "
                "or the frame. Adjust exposure before intensity range."
            ),
            wraplength=800,
            justify=tk.LEFT,
            padding=(10, 0, 10, 10),
        ).pack(fill=tk.X)

    @property
    def visible(self) -> bool:
        return self._visible

    def show(self) -> None:
        self._visible = True
        self.window.deiconify()
        self.window.lift()
        self.redraw()

    def hide(self) -> None:
        self._visible = False
        self.window.withdraw()

    def close(self) -> None:
        try:
            self.window.destroy()
        except tk.TclError:
            pass

    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def update_eye(self, index: int, packet: dict) -> None:
        self._packets[index] = packet
        if self._visible:
            self._draw_eye(index)

    def redraw(self) -> None:
        for index in range(2):
            self._draw_eye(index)

    def _draw_eye(self, index: int) -> None:
        packet = self._packets[index]
        if packet is None:
            return
        key_by_mode = {
            "Algorithm Overlay": "algorithm",
            "Dark-Pixel Mask": "mask",
            "Edges": "edges",
            "Raw IR": "raw_bgr",
        }
        image_bgr = packet[key_by_mode[self.mode_var.get()]]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb).resize(
            (400, 400), Image.Resampling.NEAREST
        )
        photo = ImageTk.PhotoImage(image_pil)
        self._photos[index] = photo
        canvas = self._canvases[index]
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self._info_vars[index].set(packet["metrics_text"])


# ── main application ───────────────────────────────────────────────────────────

class PupilDualStreamApp:
    TICK_MS = 33   # ~30 UI fps

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"Pupil Cam2 Binocular Eye Tracker v{APP_VERSION}")
        self.root.resizable(True, True)

        if _HAS_SV_TTK:
            try:
                sv_ttk.set_theme("dark")
            except tk.TclError:
                # The tracker remains functional if the optional theme asset
                # cannot be loaded by a relocated/cached Python runtime.
                pass

        self.cmap_vars    = [tk.StringVar(value="Grayscale") for _ in range(2)]
        self.detect_vars  = [tk.BooleanVar(value=True)        for _ in range(2)]
        self.flip_vars    = [tk.BooleanVar(value=False), tk.BooleanVar(value=True)]  # Eye 1 flipped by default
        self.mode_var     = tk.StringVar()
        self.exp_auto_var      = tk.BooleanVar(value=False)
        self.exp_val_var       = tk.IntVar(value=16)
        self.det_intensity_var = tk.IntVar(value=23)
        self.det_pmin_var      = tk.IntVar(value=10)
        self.det_pmax_var      = tk.IntVar(value=150)
        self.det_coarse_var    = tk.BooleanVar(value=True)
        self.assist_var        = tk.BooleanVar(value=True)
        self.show_rejected_var = tk.BooleanVar(value=False)
        self.status_var        = tk.StringVar(value="Opening cameras…")
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        self._recorder = None
        self._record_hz = tk.StringVar(value="20")
        self._record_status = tk.StringVar(value="Export: stopped (no video)")
        self._record_config = None

        self._photo:    list[Optional[ImageTk.PhotoImage]] = [None, None]
        self._prev_seq: list[int] = [0, 0]
        self._captures: list[Optional[EyeCapture]] = [None, None]
        self._uids:     list[str] = ["", ""]          # stored for mode restart
        self._mode_cb:  Optional[ttk.Combobox] = None
        self._collected_modes: list[Optional[list[str]]] = [None, None]
        self._exp_slider:        Optional[ttk.Spinbox] = None
        self._exp_val_label:     Optional[ttk.Label] = None
        self._det_intensity_lbl: Optional[ttk.Label] = None
        self._saved_mode:        Optional[str] = None
        self._detectors:    list[Optional[Detector2D]] = [None, None]
        self._det3d:        list[Optional[Detector3D]] = [None, None]
        self._det_smooth:   list[Optional[np.ndarray]] = [None, None]  # [cx,cy,rx,ry,angle]
        self._trackers = [TemporalPupilAssist(), TemporalPupilAssist()]
        self._bad_3d_frames = [0, 0]
        self._last_3d_reset = [0.0, 0.0]
        self._roi:          list[Optional[tuple]] = [None, None]       # (x,y,w,h) in camera px
        self._roi_drag:     list[Optional[tuple]] = [None, None]       # canvas drag start (x,y)
        self._gaze_view:    Optional[GazeTopDownView] = None
        self._debug_view:   Optional[DetectionDebugView] = None
        self._debug_history = [
            collections.deque(maxlen=24),
            collections.deque(maxlen=24),
        ]
        self._debug_packets: list[Optional[dict]] = [None, None]
        self._debug_tuning = False
        self._gaze_ipd_var    = tk.IntVar(value=63)
        self._gaze_mount_var  = tk.IntVar(value=600)

        self._build_ui()
        # one Detector2D + Detector3D instance per eye;
        # camera model uses a nominal focal length – refined once frame size is known
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        for i in range(2):
            self._detectors[i] = Detector2D()
            self._det3d[i]     = None   # built on first frame (needs resolution)
        self._apply_detector_props()   # apply defaults before settings load
        self._load_settings()          # overrides vars; calls _apply_detector_props again
        threading.Thread(target=self._open_cameras, daemon=True).start()
        self._tick()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        hdr = ttk.Frame(self.root, padding=(14, 10, 14, 8))
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text=f"Pupil Cam2 Binocular Eye Tracker  v{APP_VERSION}",
                  font=("Segoe UI", 15, "bold")).pack(side=tk.LEFT)
        ttk.Separator(hdr, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=14, pady=4)
        ttk.Label(hdr, textvariable=self.status_var,
                  font=("Segoe UI", 10)).pack(side=tk.LEFT)
        ttk.Separator(self.root).pack(fill=tk.X)

        # shared mode toolbar
        toolbar = ttk.Frame(self.root, padding=(14, 6, 14, 6))
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, text="Mode (both cameras):").pack(side=tk.LEFT)
        self._mode_cb = ttk.Combobox(toolbar, textvariable=self.mode_var,
                                     state="readonly", width=28)
        self._mode_cb.pack(side=tk.LEFT, padx=(8, 0))
        self._mode_cb.bind("<<ComboboxSelected>>", lambda e: self._on_mode_change())
        ttk.Button(toolbar, text="🛠 Detection Debug",
                   command=self._toggle_debug_view).pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Button(toolbar, text="▣ Gaze View",
                   command=self._toggle_gaze_view).pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Button(toolbar, text="Reset 3D models",
                   command=self._reset_tracking_models).pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Button(toolbar, text="⇄ Swap Eyes",
                   command=self._swap_eyes).pack(side=tk.RIGHT)

        # exposure row
        exp_bar = ttk.Frame(self.root, padding=(14, 2, 14, 6))
        exp_bar.pack(fill=tk.X)
        ttk.Label(exp_bar, text="Exposure:").pack(side=tk.LEFT)
        auto_exp_button = ttk.Radiobutton(
            exp_bar, text="Auto (unsupported by Cam2)",
            variable=self.exp_auto_var, value=True,
            command=self._apply_exposure,
        )
        auto_exp_button.pack(side=tk.LEFT, padx=(8, 0))
        if not CAM2_AUTO_EXPOSURE_SUPPORTED:
            auto_exp_button.configure(state="disabled")
        ttk.Radiobutton(exp_bar, text="Manual", variable=self.exp_auto_var,
                        value=False, command=self._apply_exposure).pack(side=tk.LEFT, padx=(4, 0))
        self._exp_slider = ttk.Spinbox(
            exp_bar, textvariable=self.exp_val_var,
            from_=1, to=500, increment=1, width=5,
            command=self._apply_exposure)
        self._exp_slider.bind("<Return>", lambda e: self._apply_exposure())
        self._exp_slider.pack(side=tk.LEFT, padx=(10, 4))
        self._exp_slider.configure(state="disabled")   # starts in Auto mode
        self._exp_val_label = ttk.Label(exp_bar, text="auto", width=6)
        self._exp_val_label.pack(side=tk.LEFT)

        # detector params row (mirrors Pupil Capture's tunable parameters)
        det_bar = ttk.Frame(self.root, padding=(14, 2, 14, 6))
        det_bar.pack(fill=tk.X)
        ttk.Label(det_bar, text="Detector:").pack(side=tk.LEFT)
        ttk.Checkbutton(det_bar, text="Coarse detection",
                        variable=self.det_coarse_var,
                        command=self._apply_detector_props).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(det_bar, text="Intensity range:").pack(side=tk.LEFT, padx=(14, 0))
        ttk.Scale(det_bar, variable=self.det_intensity_var, from_=1, to=50,
                  orient=tk.HORIZONTAL, length=110,
                  command=lambda v: self._apply_detector_props()).pack(side=tk.LEFT, padx=(4, 2))
        self._det_intensity_lbl = ttk.Label(det_bar, text="23", width=3)
        self._det_intensity_lbl.pack(side=tk.LEFT)
        ttk.Label(det_bar, text="Pupil px:").pack(side=tk.LEFT, padx=(14, 0))
        ttk.Spinbox(det_bar, textvariable=self.det_pmin_var, from_=5, to=200, width=4,
                    command=self._apply_detector_props).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(det_bar, text="–").pack(side=tk.LEFT, padx=2)
        ttk.Spinbox(det_bar, textvariable=self.det_pmax_var, from_=20, to=400, width=4,
                    command=self._apply_detector_props).pack(side=tk.LEFT)
        ttk.Checkbutton(det_bar, text="Assist tracking",
                        variable=self.assist_var).pack(side=tk.LEFT, padx=(14, 0))
        ttk.Checkbutton(det_bar, text="Show rejected",
                        variable=self.show_rejected_var).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Separator(self.root).pack(fill=tk.X)

        body = ttk.Frame(self.root, padding=10)
        record_bar = ttk.Frame(self.root, padding=(14, 2, 14, 4))
        record_bar.pack(fill=tk.X)
        ttk.Button(record_bar, text="Start / Stop export", command=self._toggle_recording).pack(side=tk.LEFT)
        ttk.Label(record_bar, text="Max Hz / eye:").pack(side=tk.LEFT, padx=(8, 4))
        ttk.Combobox(record_bar, textvariable=self._record_hz, values=(10, 20, 30),
                     width=4, state="readonly").pack(side=tk.LEFT)
        ttk.Button(record_bar, text="Mark event", command=self._mark_recording).pack(side=tk.LEFT, padx=8)
        ttk.Label(record_bar, textvariable=self._record_status, width=48).pack(side=tk.LEFT)
        body.pack(fill=tk.BOTH, expand=True)

        self._canvases:  list[tk.Canvas]    = []
        self._info_vars: list[tk.StringVar] = []
        self._det_info_vars: list[tk.StringVar] = []

        for col, label in enumerate(EYE_LABELS):
            frm = ttk.LabelFrame(body, text=label, padding=8)
            frm.grid(row=0, column=col, padx=(0, 10 if col == 0 else 0), sticky="nw")

            canvas = tk.Canvas(frm, width=PREVIEW_W, height=PREVIEW_H,
                               bg="#0d1117", highlightthickness=0)
            canvas.pack()
            self._canvases.append(canvas)
            canvas.bind("<ButtonPress-1>",  lambda e, i=col: self._on_roi_press(i, e))
            canvas.bind("<B1-Motion>",       lambda e, i=col: self._on_roi_drag_cb(i, e))
            canvas.bind("<ButtonRelease-1>", lambda e, i=col: self._on_roi_release(i, e))
            canvas.bind("<ButtonPress-3>",   lambda e, i=col: self._on_roi_clear(i))

            info_var = tk.StringVar(value="Waiting…")
            ttk.Label(frm, textvariable=info_var,
                      font=("Consolas", 9)).pack(anchor=tk.W, pady=(4, 0))
            self._info_vars.append(info_var)

            # controls row
            ctrl = ttk.Frame(frm)
            ctrl.pack(fill=tk.X, pady=(4, 0))
            ttk.Label(ctrl, text="Color:").pack(side=tk.LEFT)
            ttk.Combobox(ctrl, textvariable=self.cmap_vars[col],
                         values=list(COLORMAPS.keys()),
                         state="readonly", width=11).pack(side=tk.LEFT, padx=(4, 12))
            ttk.Checkbutton(ctrl, text="Detect Pupil",
                            variable=self.detect_vars[col],
                            command=lambda i=col: self._on_detect_changed(i)).pack(side=tk.LEFT)
            ttk.Checkbutton(ctrl, text="Flip H",
                            variable=self.flip_vars[col],
                            command=lambda i=col: self._on_flip_changed(i)).pack(side=tk.LEFT, padx=(6, 0))
            ttk.Button(ctrl, text="⌧ ROI",
                       command=lambda i=col: self._on_roi_clear(i)).pack(side=tk.LEFT, padx=(8, 0))

            det_info_var = tk.StringVar(value="")
            status_box = ttk.Frame(frm, width=PREVIEW_W, height=72)
            status_box.pack(anchor=tk.W)
            status_box.pack_propagate(False)
            ttk.Label(status_box, textvariable=det_info_var, wraplength=PREVIEW_W - 8,
                      justify=tk.LEFT, anchor="nw",
                      font=("Consolas", 8), foreground="#44ff88").pack(fill=tk.BOTH, expand=True)
            self._det_info_vars.append(det_info_var)

        # Gaze top-down panel as embedded third column (hidden by default)
        self._gaze_view = GazeTopDownView(
            body, self._gaze_ipd_var, self._gaze_mount_var)

    # ── ROI drawing ───────────────────────────────────────────────────────────

    def _canvas_to_img(self, idx: int, cx: float, cy: float) -> tuple:
        """Convert canvas pixel coordinates to camera image pixel coordinates."""
        cap = self._captures[idx]
        fw = cap.width  if (cap and cap.width)  else PREVIEW_W
        fh = cap.height if (cap and cap.height) else PREVIEW_H
        scale = min(PREVIEW_W / fw, PREVIEW_H / fh)
        ox = (PREVIEW_W - round(fw * scale)) // 2
        oy = (PREVIEW_H - round(fh * scale)) // 2
        ix = int(round((cx - ox) / scale))
        iy = int(round((cy - oy) / scale))
        return (max(0, min(fw - 1, ix)), max(0, min(fh - 1, iy)))

    def _on_roi_press(self, idx: int, event: tk.Event) -> None:
        self._roi_drag[idx] = (event.x, event.y)

    def _on_roi_drag_cb(self, idx: int, event: tk.Event) -> None:
        start = self._roi_drag[idx]
        if start is None:
            return
        canvas = self._canvases[idx]
        canvas.delete("roi_drag")
        x0, y0 = start
        canvas.create_rectangle(
            min(x0, event.x), min(y0, event.y),
            max(x0, event.x), max(y0, event.y),
            outline="#00dcff", width=2, dash=(5, 3), tags="roi_drag")

    def _on_roi_release(self, idx: int, event: tk.Event) -> None:
        start = self._roi_drag[idx]
        self._roi_drag[idx] = None
        self._canvases[idx].delete("roi_drag")
        if start is None:
            return
        ix0, iy0 = self._canvas_to_img(idx, start[0], start[1])
        ix1, iy1 = self._canvas_to_img(idx, event.x, event.y)
        rw = abs(ix1 - ix0)
        rh = abs(iy1 - iy0)
        if rw < 5 or rh < 5:
            return  # ignore tiny accidental clicks
        self._roi[idx]        = (min(ix0, ix1), min(iy0, iy1), rw, rh)
        self._det_smooth[idx] = None  # reset EMA – coordinate context changed
        self._trackers[idx].reset()
        self._det3d[idx] = None

    def _on_roi_clear(self, idx: int) -> None:
        self._roi[idx]        = None
        self._det_smooth[idx] = None
        self._trackers[idx].reset()
        self._det3d[idx] = None

    # ── camera open (background thread) ──────────────────────────────────────

    def _open_cameras(self) -> None:
        uids = _find_pupil_uids()
        if len(uids) != 2:
            msg = (f"Found {len(uids)} matching Pupil Cam2 camera(s) – need exactly 2. "
                   "Check VID_0C45/PID_64AB parent devices and libusbK.")
            self.root.after(0, lambda: self.status_var.set(msg))
            return

        self._uids = list(uids[:2])
        for idx, uid in enumerate(self._uids):
            self._start_eye(idx, uid, want_mode=None)
        

        self.root.after(0, lambda: self.status_var.set(
            "Streaming both Cam2 sensors – 400×400 direct UVC"))

    def _start_eye(self, idx: int, uid: str, want_mode: Optional[str]) -> None:
        """(Re)start a single eye capture thread, optionally requesting a mode."""
        old = self._captures[idx]
        if old is not None:
            old.stop()
            old.join(timeout=2.0)

        def _modes_ready(labels: list[str], _idx: int = idx) -> None:
            self.root.after(0, lambda: self._populate_mode_cb(_idx, labels))

        def _exp_ready(mn: int, mx: int, cur: int) -> None:
            self.root.after(0, lambda: self._on_exposure_ready(mn, mx, cur))

        self._roi[idx]        = None
        self._det_smooth[idx] = None
        self._det3d[idx]      = None
        self._trackers[idx].reset()
        self._bad_3d_frames[idx] = 0
        cap = EyeCapture(eye_id=idx, uid=uid,
                         want_mode=want_mode,
                         on_modes_ready=_modes_ready,
                         on_exposure_ready=_exp_ready)
        cap.start()
        self._captures[idx] = cap

    def _populate_mode_cb(self, idx: int, labels: list[str]) -> None:
        self._collected_modes[idx] = labels
        # wait until both cameras have reported their modes
        if any(m is None for m in self._collected_modes):
            return
        # intersect, preserving eye 0 order
        both = [l for l in self._collected_modes[0] if l in self._collected_modes[1]]
        if not both:
            both = self._collected_modes[0]
        cb = self._mode_cb
        if cb is None:
            return
        cb.configure(values=both)
        cap0 = self._captures[0]
        current = cap0.active_mode_label if cap0 else ""
        preferred = self._saved_mode if (self._saved_mode and self._saved_mode in both) else None
        selected  = preferred or (current if current in both else (both[0] if both else ""))
        self.mode_var.set(selected)
        if preferred and preferred != current:
            self._saved_mode = None   # consume once; prevent recursive trigger
            self._on_mode_change()

    def _on_exposure_ready(self, exp_min: int, exp_max: int, current: int) -> None:
        if self._exp_slider is None:
            return
        self._exp_slider.configure(from_=exp_min, to=exp_max)
        if not self.exp_auto_var.get():
            self.exp_val_var.set(max(exp_min, min(exp_max, current)))
        self._apply_exposure()   # always push current mode (auto or manual) to camera on open

    def _apply_exposure(self) -> None:
        auto = bool(self.exp_auto_var.get() and CAM2_AUTO_EXPOSURE_SUPPORTED)
        if self.exp_auto_var.get() != auto:
            self.exp_auto_var.set(auto)
        val  = int(self.exp_val_var.get())
        if self._exp_slider is not None:
            self._exp_slider.configure(state="disabled" if auto else "normal")
        if self._exp_val_label is not None:
            self._exp_val_label.configure(text="auto" if auto else str(val))
        for cap in self._captures:
            if cap:
                cap.apply_exposure(auto, val)

    def _apply_detector_props(self) -> None:
        """Push current UI detector parameters to both Detector2D instances."""
        try:
            irange = int(self.det_intensity_var.get())
            pmin   = int(self.det_pmin_var.get())
            pmax   = int(self.det_pmax_var.get())
        except (tk.TclError, ValueError):
            return
        if self._det_intensity_lbl is not None:
            self._det_intensity_lbl.configure(text=str(irange))
        props = {
            "coarse_detection":  bool(self.det_coarse_var.get()),
            "coarse_filter_min": 100,
            "coarse_filter_max": 8000,
            "intensity_range":   irange,
            "pupil_size_min":    pmin,
            "pupil_size_max":    pmax,
        }
        for det in self._detectors:
            if det is not None:
                det.update_properties(props)

    def _load_settings(self) -> None:
        """Restore UI state from settings file (silently ignores missing/corrupt file)."""
        try:
            load_path = SETTINGS_PATH if SETTINGS_PATH.exists() else LEGACY_SETTINGS_PATH
            with open(load_path, encoding="utf-8") as f:
                s = json.load(f)
        except Exception:
            return
        try:
            if "exp_auto" in s:
                self.exp_auto_var.set(
                    bool(s["exp_auto"]) and CAM2_AUTO_EXPOSURE_SUPPORTED
                )
            if "exp_val" in s:
                self.exp_val_var.set(int(s["exp_val"]))
            if "cmap" in s:
                for i, v in enumerate(s["cmap"][:2]):
                    if v in COLORMAPS:
                        self.cmap_vars[i].set(v)
            if "detect" in s:
                for i, v in enumerate(s["detect"][:2]):
                    self.detect_vars[i].set(bool(v))
            if "flip" in s:
                for i, v in enumerate(s["flip"][:2]):
                    self.flip_vars[i].set(bool(v))
            if "intensity_range" in s:
                self.det_intensity_var.set(int(s["intensity_range"]))
            if "pupil_size_min" in s:
                self.det_pmin_var.set(int(s["pupil_size_min"]))
            if "pupil_size_max" in s:
                self.det_pmax_var.set(int(s["pupil_size_max"]))
            if "coarse_detection" in s:
                self.det_coarse_var.set(bool(s["coarse_detection"]))
            if "assist_tracking" in s:
                self.assist_var.set(bool(s["assist_tracking"]))
            if "show_rejected" in s:
                self.show_rejected_var.set(bool(s["show_rejected"]))
            if "mode" in s:
                self._saved_mode = s["mode"]
            if "gaze_ipd" in s:
                self._gaze_ipd_var.set(int(s["gaze_ipd"]))
            if "gaze_mount" in s:
                self._gaze_mount_var.set(int(s["gaze_mount"]))
            if "gaze_cam_angle" in s:
                self._gaze_view.cam_angle_var.set(int(s["gaze_cam_angle"]))
            if "gaze_toe_in" in s:
                self._gaze_view.screen_tilt_var.set(int(s["gaze_toe_in"]))
            if "gaze_screen_depth" in s:
                self._gaze_view.screen_depth_var.set(int(s["gaze_screen_depth"]))
        except Exception:
            pass
        self._apply_detector_props()
        self._apply_exposure()

    def _toggle_gaze_view(self) -> None:
        if self._gaze_view is None:
            return
        if self._gaze_view._visible:
            self._gaze_view.hide()
        else:
            self._gaze_view.show()

    def _reset_tracking_models(self) -> None:
        """Clear both temporal trackers and rebuild both 3D models."""
        if self._recorder:
            try:
                self._recorder.event("model_reset", "manual reset of both eyes")
            except OSError as exc:
                self._recording_failed(exc)
        for idx in range(2):
            self._trackers[idx].reset()
            self._det_smooth[idx] = None
            self._det3d[idx] = None
            self._bad_3d_frames[idx] = 0
            self._last_3d_reset[idx] = time.monotonic()
            self._det_info_vars[idx].set("tracking model reset")
            if self._gaze_view:
                self._gaze_view.invalidate(idx)

    def _on_detect_changed(self, idx: int) -> None:
        self._trackers[idx].reset()
        self._det_smooth[idx] = None
        self._det3d[idx] = None
        self._bad_3d_frames[idx] = 0
        if not self.detect_vars[idx].get() and self._gaze_view:
            self._gaze_view.invalidate(idx)

    def _toggle_debug_view(self) -> None:
        if self._debug_view is None:
            self._debug_view = DetectionDebugView(self.root, self)
        if self._debug_view.visible:
            self._debug_view.hide()
        else:
            self._debug_view.show()

    def _debug_visible(self) -> bool:
        return self._debug_view is not None and self._debug_view.visible

    def _build_debug_packet(
            self, idx: int, frame: np.ndarray,
            algorithm_bgr: Optional[np.ndarray],
            result_2d: Optional[dict], result_3d: Optional[dict]) -> dict:
        """Build vendor-style algorithm, threshold, edge, and metric views."""
        height, width = frame.shape
        roi = self._roi[idx]
        if roi is None:
            x, y, rw, rh = 0, 0, width, height
        else:
            x = max(0, min(width - 1, int(roi[0])))
            y = max(0, min(height - 1, int(roi[1])))
            rw = max(1, min(width - x, int(roi[2])))
            rh = max(1, min(height - y, int(roi[3])))

        raw_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        algorithm = raw_bgr.copy() if algorithm_bgr is None else algorithm_bgr.copy()
        region = frame[y:y + rh, x:x + rw]
        p05, p50, p95 = [float(v) for v in np.percentile(region, (5, 50, 95))]
        threshold_value = min(255.0, p05 + float(self.det_intensity_var.get()))
        local_dark = region <= threshold_value
        dark_pixels = np.zeros_like(frame, dtype=bool)
        dark_pixels[y:y + rh, x:x + rw] = local_dark

        mask_bgr = (raw_bgr.astype(np.float32) * 0.30).astype(np.uint8)
        mask_bgr[dark_pixels] = (255, 80, 0)  # blue, like algorithm-mode candidates
        blurred = cv2.GaussianBlur(frame, (5, 5), 1.0)
        edges_gray = cv2.Canny(blurred, 25, 75)
        edges_bgr = cv2.cvtColor(edges_gray, cv2.COLOR_GRAY2BGR)

        confidence_2d = float((result_2d or {}).get("confidence", 0.0))
        confidence_3d = float((result_3d or {}).get("model_confidence", 0.0))
        tracking_source = str((result_2d or {}).get("tracking_source", ""))
        tracking_reason = str((result_2d or {}).get("tracking_reject_reason", ""))
        tracking_contrast = float((result_2d or {}).get("tracking_contrast", 0.0))
        diameter = float((result_2d or {}).get("diameter", 0.0))
        ellipse = (result_2d or {}).get("ellipse")
        if ellipse:
            raw_axes = [float(v) for v in ellipse.get("axes", (0.0, 0.0))]
            if diameter <= 0.0 or not raw_axes or max(raw_axes) <= 0.0:
                ellipse = None
        if ellipse:
            center = tuple(int(round(float(v))) for v in ellipse["center"])
            axes = tuple(max(1, int(round(float(v) / 2))) for v in ellipse["axes"])
            angle = float(ellipse["angle"])
            for debug_image in (raw_bgr, mask_bgr, edges_bgr):
                cv2.ellipse(debug_image, center, axes, angle,
                            0, 360, (0, 215, 255), 2)
                cv2.circle(debug_image, center, 3, (0, 215, 255), -1)

        for debug_image in (raw_bgr, algorithm, mask_bgr, edges_bgr):
            cv2.rectangle(debug_image, (x, y), (x + rw - 1, y + rh - 1),
                          (255, 255, 0), 1)

        dark_coverage = float(np.mean(local_dark)) if local_dark.size else 0.0
        if confidence_2d >= 0.75:
            quality = "GOOD"
        elif confidence_2d >= 0.45:
            quality = "CHECK"
        else:
            quality = "POOR"
        metrics_text = (
            f"2D conf={confidence_2d:.3f}  3D model={confidence_3d:.3f}  "
            f"diameter={diameter:.1f}px  [{quality}]\n"
            f"gate={tracking_source or 'n/a'}  contrast={tracking_contrast:.1f}  "
            f"reason={tracking_reason or 'n/a'}\n"
            f"IR p05/p50/p95={p05:.0f}/{p50:.0f}/{p95:.0f}  "
            f"dark threshold={threshold_value:.0f}  coverage={dark_coverage:.1%}\n"
            f"intensity={int(self.det_intensity_var.get())}  "
            f"pupil px={int(self.det_pmin_var.get())}–{int(self.det_pmax_var.get())}  "
            f"ROI=({x},{y},{rw},{rh})"
        )

        ellipse_data = None
        if ellipse:
            ellipse_data = {
                "center": [float(v) for v in ellipse["center"]],
                "axes": [float(v) for v in ellipse["axes"]],
                "angle": float(ellipse["angle"]),
            }
        return {
            "raw_bgr": raw_bgr,
            "algorithm": algorithm,
            "mask": mask_bgr,
            "edges": edges_bgr,
            "metrics_text": metrics_text,
            "data": {
                "eye": idx,
                "confidence_2d": confidence_2d,
                "model_confidence_3d": confidence_3d,
                "tracking_source": tracking_source,
                "tracking_reason": tracking_reason,
                "tracking_contrast": tracking_contrast,
                "diameter_px": diameter,
                "ellipse": ellipse_data,
                "ir_percentiles": {"p05": p05, "p50": p50, "p95": p95},
                "dark_threshold": threshold_value,
                "dark_coverage": dark_coverage,
                "roi_xywh": [x, y, rw, rh],
                "parameters": {
                    "intensity_range": int(self.det_intensity_var.get()),
                    "pupil_size_min": int(self.det_pmin_var.get()),
                    "pupil_size_max": int(self.det_pmax_var.get()),
                    "coarse_detection": bool(self.det_coarse_var.get()),
                    "exposure": int(self.exp_val_var.get()),
                },
            },
        }

    def _auto_tune_detection(self) -> None:
        """Sweep 2D parameters over recent frames and apply a stable candidate."""
        if self._debug_tuning:
            return
        histories = [[frame.copy() for frame in history]
                     for history in self._debug_history]
        if any(len(history) < 6 for history in histories):
            if self._debug_view:
                self._debug_view.set_status(
                    "Need at least 6 recent frames per eye. Keep both eyes visible briefly."
                )
            return

        rois = list(self._roi)
        pupil_min = int(self.det_pmin_var.get())
        pupil_max = int(self.det_pmax_var.get())
        self._debug_tuning = True
        if self._debug_view:
            self._debug_view.set_status(
                "Auto Tune is evaluating intensity/coarse settings on recent frames…"
            )

        def worker() -> None:
            try:
                best = None
                for intensity in range(5, 61, 5):
                    for coarse in (True, False):
                        eye_scores = []
                        all_valid_diameters = []
                        valid_ratios = []
                        median_confidences = []
                        for eye_index, eye_frames in enumerate(histories):
                            detector = Detector2D()
                            detector.update_properties({
                                "coarse_detection": coarse,
                                "coarse_filter_min": 100,
                                "coarse_filter_max": 8000,
                                "intensity_range": intensity,
                                "pupil_size_min": pupil_min,
                                "pupil_size_max": pupil_max,
                            })
                            confidences = []
                            valid_centers = []
                            valid_diameters = []
                            for frame in eye_frames[-10:]:
                                roi = rois[eye_index]
                                if roi is None:
                                    detector_roi = Roi.from_shape(frame.shape)
                                else:
                                    x, y, rw, rh = roi
                                    detector_roi = Roi(x, y, x + rw, y + rh)
                                result = detector.detect(frame, roi=detector_roi)
                                confidence = float(result.get("confidence", 0.0))
                                confidences.append(confidence)
                                ellipse = result.get("ellipse")
                                if confidence >= 0.55 and ellipse:
                                    valid_centers.append(
                                        [float(v) for v in ellipse["center"]]
                                    )
                                    valid_diameters.append(
                                        float(result.get("diameter", 0.0))
                                    )
                            valid_ratio = len(valid_centers) / max(1, len(confidences))
                            median_conf = float(np.median(confidences))
                            if len(valid_centers) >= 3:
                                center_std = float(np.mean(np.std(valid_centers, axis=0)))
                                diameter_mean = max(1.0, float(np.mean(valid_diameters)))
                                diameter_cv = float(np.std(valid_diameters)) / diameter_mean
                                stability = max(
                                    0.0, 1.0 - center_std / 35.0 - diameter_cv
                                )
                            else:
                                stability = 0.0
                            score = (0.55 * median_conf
                                     + 0.35 * valid_ratio
                                     + 0.10 * stability)
                            eye_scores.append(score)
                            valid_ratios.append(valid_ratio)
                            median_confidences.append(median_conf)
                            all_valid_diameters.extend(valid_diameters)

                        combined_score = min(eye_scores) + 0.20 * float(np.mean(eye_scores))
                        candidate = {
                            "score": combined_score,
                            "intensity": intensity,
                            "coarse": coarse,
                            "valid_ratios": valid_ratios,
                            "median_confidences": median_confidences,
                            "diameters": all_valid_diameters,
                        }
                        if best is None or candidate["score"] > best["score"]:
                            best = candidate
                self.root.after(0, lambda: self._finish_auto_tune(best))
            except Exception as exc:
                message = str(exc)
                self.root.after(
                    0, lambda message=message: self._finish_auto_tune(None, message)
                )

        threading.Thread(target=worker, daemon=True,
                         name="pupil-auto-tune").start()

    def _finish_auto_tune(self, best: Optional[dict], error: str = "") -> None:
        self._debug_tuning = False
        if self._debug_view is None:
            return
        if error or best is None:
            self._debug_view.set_status(f"Auto Tune failed: {error or 'no result'}")
            return
        if (min(best["valid_ratios"]) < 0.30
                or min(best["median_confidences"]) < 0.35):
            self._debug_view.set_status(
                "Auto Tune found no trustworthy binocular setting. Adjust exposure/ROI "
                "and verify the dark-pixel mask before trying again. No values were changed."
            )
            return

        self.det_intensity_var.set(int(best["intensity"]))
        self.det_coarse_var.set(bool(best["coarse"]))
        if best["diameters"]:
            diameter = float(np.median(best["diameters"]))
            suggested_min = max(5, int(round(diameter * 0.55)))
            suggested_max = min(400, int(round(diameter * 1.75)))
            if suggested_min < suggested_max:
                self.det_pmin_var.set(suggested_min)
                self.det_pmax_var.set(suggested_max)
        self._apply_detector_props()
        self._debug_view.set_status(
            f"Applied intensity={best['intensity']}, coarse={best['coarse']}, "
            f"pupil px={self.det_pmin_var.get()}–{self.det_pmax_var.get()}. "
            "Visually confirm the Algorithm Overlay before recording."
        )

    def _apply_suggested_roi(self, idx: int) -> None:
        packet = self._debug_packets[idx]
        if packet is None or packet["data"]["ellipse"] is None:
            if self._debug_view:
                self._debug_view.set_status(
                    f"Eye {idx}: no pupil candidate is available for an ROI suggestion."
                )
            return
        data = packet["data"]
        if data["confidence_2d"] < 0.45:
            if self._debug_view:
                self._debug_view.set_status(
                    f"Eye {idx}: candidate confidence is too low; ROI was not changed."
                )
            return
        ellipse = data["ellipse"]
        cx, cy = ellipse["center"]
        axis_x, axis_y = ellipse["axes"]
        frame = self._captures[idx].get_frame() if self._captures[idx] else None
        if frame is None:
            return
        height, width = frame.shape
        half_width = max(80.0, axis_x * 2.2)
        half_height = max(80.0, axis_y * 2.2)
        x0 = max(0, int(round(cx - half_width)))
        y0 = max(0, int(round(cy - half_height)))
        x1 = min(width, int(round(cx + half_width)))
        y1 = min(height, int(round(cy + half_height)))
        self._roi[idx] = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
        self._det_smooth[idx] = None
        self._det3d[idx] = None
        self._trackers[idx].reset()
        self._bad_3d_frames[idx] = 0
        if self._debug_view:
            self._debug_view.set_status(
                f"Eye {idx}: suggested ROI applied. Look to all gaze extremes and enlarge "
                "manually if the pupil leaves the cyan box."
            )

    def _save_debug_pack(self) -> None:
        if any(packet is None for packet in self._debug_packets):
            if self._debug_view:
                self._debug_view.set_status(
                    "Both eyes need current debug data before saving."
                )
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = DEBUG_SAVE_DIR / stamp
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "created_at": stamp,
            "mode": self.mode_var.get(),
            "eyes": [],
        }
        for idx, packet in enumerate(self._debug_packets):
            for key in ("raw_bgr", "algorithm", "mask", "edges"):
                cv2.imencode(".png", packet[key])[1].tofile(str(output_dir / f"eye{idx}_{key}.png"))
            history = list(self._debug_history[idx])[-8:]
            for frame_index, frame in enumerate(history):
                cv2.imencode(".png", frame)[1].tofile(
                    str(output_dir / f"eye{idx}_history_{frame_index:02d}.png"))
            metadata["eyes"].append(packet["data"])
        with open(output_dir / "detection_metrics.json", "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)
        if self._debug_view:
            self._debug_view.set_status(f"Debug pack saved: {output_dir}")

    def _save_settings(self) -> None:
        """Persist current UI state to settings file."""
        try:
            s = {
                "mode":             self.mode_var.get(),
                "exp_auto":         bool(self.exp_auto_var.get()),
                "exp_val":          int(self.exp_val_var.get()),
                "cmap":             [v.get() for v in self.cmap_vars],
                "detect":           [bool(v.get()) for v in self.detect_vars],
                "flip":             [bool(v.get()) for v in self.flip_vars],
                "intensity_range":  int(self.det_intensity_var.get()),
                "pupil_size_min":   int(self.det_pmin_var.get()),
                "pupil_size_max":   int(self.det_pmax_var.get()),
                "coarse_detection": bool(self.det_coarse_var.get()),
                "assist_tracking":  bool(self.assist_var.get()),
                "show_rejected":    bool(self.show_rejected_var.get()),
                "gaze_ipd":         int(self._gaze_ipd_var.get()),
                "gaze_mount":       int(self._gaze_mount_var.get()),
                "gaze_cam_angle":   int(self._gaze_view.cam_angle_var.get()),
                "gaze_toe_in":      int(self._gaze_view.screen_tilt_var.get()),
                "gaze_screen_depth": int(self._gaze_view.screen_depth_var.get()),
            }
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2)
        except Exception:
            pass

    def _swap_eyes(self) -> None:
        self._captures.reverse()
        self._uids.reverse()
        self._collected_modes.reverse()
        self._prev_seq.reverse()
        self._photo.reverse()
        self._roi.reverse()
        self._det_smooth.reverse()
        self._detectors.reverse()
        self._det3d.reverse()
        self._trackers.reverse()
        self._bad_3d_frames.reverse()
        self._last_3d_reset.reverse()
        self._debug_history.reverse()
        self._debug_packets.reverse()
        for v in self.flip_vars:
            v.set(not v.get())   # swap also flips which eye needs horizontal mirror
        if self._gaze_view:
            self._gaze_view._gaze.reverse()
            self._gaze_view._gaze_updated.reverse()
        # Swap changes the image coordinate system. Old 3D fits are invalid.
        for idx in range(2):
            self._on_flip_changed(idx)

    def _on_mode_change(self) -> None:
        want = self.mode_var.get()
        if not want:
            return
        self.status_var.set(f"Switching both cameras to {want}…")
        self._collected_modes = [None, None]
        def _switch():
            for idx, uid in enumerate(self._uids):
                if uid:
                    self._start_eye(idx, uid, want)
        threading.Thread(target=_switch, daemon=True).start()

    # ── tick ──────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        if self._recorder:
            try:
                config = self._recording_metadata()
                if config != self._record_config:
                    self._recorder.event("configuration", json.dumps(config, ensure_ascii=False))
                    self._record_config = config
            except (tk.TclError, ValueError):
                pass  # A numeric field may be temporarily empty during editing.
            except OSError as exc:
                self._recording_failed(exc)
        for idx, cap in enumerate(self._captures):
            if cap is None:
                self._export_sample(idx, time.monotonic(), "NO_CAMERA")
                continue
            if cap.error:
                self._export_sample(idx, time.monotonic(), "CAMERA_ERROR")
                self._info_vars[idx].set(cap.error)
                if self._gaze_view:
                    self._gaze_view.invalidate(idx)
                continue

            frame, seq, frame_timestamp = cap.get_latest()
            if frame is None or time.monotonic() - frame_timestamp > 0.25:
                self._export_sample(idx, time.monotonic(), "STALE")
                if self._gaze_view:
                    self._gaze_view.invalidate(idx)
                self._det_info_vars[idx].set("STALE FRAME / gaze paused")
                continue
            if seq != self._prev_seq[idx]:
                self._prev_seq[idx] = seq
                if frame is not None:
                    self._render(idx, frame, frame_timestamp or time.monotonic())

            w, h, fps = cap.width, cap.height, cap.get_fps()
            if w and h:
                self._info_vars[idx].set(f"{w}×{h}  |  {fps:5.1f} fps  |  frame {seq}")
            else:
                self._info_vars[idx].set("Waiting for frames…")

        self.root.after(self.TICK_MS, self._tick)

    # ── render ────────────────────────────────────────────────────────────────

    def _on_flip_changed(self, idx: int) -> None:
        """Reset 3D model and smooth state when flip is toggled – old model is invalid."""
        self._det_smooth[idx] = None
        self._det3d[idx]      = None
        self._roi[idx]        = None
        self._trackers[idx].reset()
        self._bad_3d_frames[idx] = 0
        if self._gaze_view:
            self._gaze_view.invalidate(idx)

    def _render(self, idx: int, frame: np.ndarray,
                frame_timestamp: Optional[float] = None) -> None:
        if frame_timestamp is None:
            frame_timestamp = time.monotonic()
        if self.flip_vars[idx].get():
            frame = np.ascontiguousarray(frame[:, ::-1])
        fh_src, fw_src = frame.shape
        roi = self._roi[idx]

        cmap = COLORMAPS[self.cmap_vars[idx].get()]
        if cmap is not None:
            rgb = cv2.cvtColor(cv2.applyColorMap(frame, cmap), cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        debug_result_2d = None
        debug_result_3d = None
        algorithm_bgr = (
            cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if self._debug_visible() else None
        )

        if self.detect_vars[idx].get():
            det2 = self._detectors[idx]
            if det2 is not None:
                try:
                    # build Detector3D once (full-frame resolution)
                    if self._det3d[idx] is None:
                        cam = CameraModel(focal_length=_cam2_focal_length(fw_src, fh_src),
                                          resolution=(fw_src, fh_src))
                        self._det3d[idx] = Detector3D(cam)

                    # ── 2-D detection on full frame with native ROI hint ───────
                    # Using Roi() keeps all result coordinates in full-frame space
                    # (no manual crop/offset that would corrupt the outline points
                    # that pye3d uses for 3D fitting).  Detect on the raw frame –
                    # CLAHE distorts the intensity distribution the algorithm needs.
                    if roi is not None:
                        rx = max(0, min(fw_src - 1, roi[0]))
                        ry = max(0, min(fh_src - 1, roi[1]))
                        rw = max(1, min(fw_src - rx, roi[2]))
                        rh = max(1, min(fh_src - ry, roi[3]))
                        det_roi = Roi(rx, ry, rx + rw, ry + rh)
                    else:
                        det_roi = Roi.from_shape(frame.shape)

                    r2 = det2.detect(
                        frame, color_img=algorithm_bgr, roi=det_roi
                    )
                    r2 = self._trackers[idx].refine_primary(
                        r2, frame, roi, int(self.det_pmin_var.get()), int(self.det_pmax_var.get()))
                    r2["timestamp"] = frame_timestamp

                    tracker = self._trackers[idx]
                    decision = tracker.process(
                        r2, frame, roi,
                        int(self.det_pmin_var.get()),
                        int(self.det_pmax_var.get()),
                        frame_timestamp,
                        bool(self.assist_var.get()),
                    )
                    r2["tracking_source"] = decision.source
                    r2["tracking_reject_reason"] = decision.reason
                    r2["tracking_contrast"] = decision.contrast

                    # Only raw, visibility-validated primary ellipses enter pye3d.
                    r2_for_3d = dict(r2)

                    # ── 3-D detection ─────────────────────────────────────────
                    r3 = {}
                    if decision.primary_accepted and tracker.gaze_allowed:
                        with _suppress_stdout():
                            r3 = self._det3d[idx].update_and_detect(r2_for_3d, frame)
                    debug_result_2d = r2
                    debug_result_3d = r3

                    conf2 = float(r2.get("confidence", 0))
                    conf3 = float(r3.get("model_confidence", 0))

                    warm_elapsed = 0.0
                    if tracker.first_primary_ts is not None:
                        warm_elapsed = max(0.0, frame_timestamp - tracker.first_primary_ts)
                    theta_v = r3.get("theta")
                    phi_v = r3.get("phi")
                    model_ready = (
                        tracker.gaze_allowed and decision.primary_accepted
                        and warm_elapsed >= 5.0 and tracker.primary_count >= 60 and conf3 > 0.5
                        and float(r3.get("confidence", 0.0)) >= 0.70
                        and theta_v is not None and phi_v is not None
                        and np.all(np.isfinite([float(theta_v), float(phi_v)]))
                    )
                    self._export_sample(
                        idx, frame_timestamp, tracker.visibility,
                        ellipse=_normalise_ellipse(r2.get("ellipse")) if decision.primary_accepted else None,
                        gaze=_pye3d_direction(float(theta_v), float(phi_v)) if model_ready else None,
                        confidence=conf2, boundary=tracker.edge_support, model_confidence=conf3)
                    if model_ready:
                        self._bad_3d_frames[idx] = 0
                    elif decision.primary_accepted and warm_elapsed >= 5.0:
                        self._bad_3d_frames[idx] += 1

                    raw_ellipse = _normalise_ellipse(r2.get("ellipse"))
                    if (self.show_rejected_var.get()
                            and decision.source != "PRIMARY"
                            and raw_ellipse is not None):
                        rejected_center = tuple(
                            int(round(value)) for value in raw_ellipse["center"]
                        )
                        rejected_axes = tuple(
                            max(1, int(round(value / 2.0)))
                            for value in raw_ellipse["axes"]
                        )
                        cv2.ellipse(
                            rgb, rejected_center, rejected_axes,
                            raw_ellipse["angle"], 0, 360, (255, 70, 70), 1,
                        )

                    # ── draw pupil ellipse ────────────────────────────────────
                    s = decision.ellipse
                    if s is not None:
                        cx2, cy2 = s["center"]
                        axis0, axis1 = s["axes"]
                        ang2 = s["angle"]
                        center = (int(round(cx2)), int(round(cy2)))
                        axes = (max(1, int(round(axis0 / 2.0))),
                                max(1, int(round(axis1 / 2.0))))
                        color = {
                            "PRIMARY": (30, 255, 80),
                            "ASSIST": (0, 220, 255),
                            "PREDICT": (255, 170, 0),
                        }.get(decision.source, (255, 70, 70))
                        thickness = 2 if decision.source != "PREDICT" else 1
                        cv2.ellipse(rgb, center, axes, ang2, 0, 360, color, thickness)
                        cv2.circle(rgb, center, 3, color, -1)

                        # ── gaze direction arrow ──────────────────────────────
                        # Spherical gaze vector in camera space:
                        #   gx = sin(theta)*cos(phi)  → rightward in image
                        #   gy = cos(theta)  → downward in image
                        if model_ready and decision.source != "LOST":
                            th = float(theta_v)
                            ph = float(phi_v)
                            arrow_len = max(fw_src, fh_src) * 0.40
                            gx, gy, _ = _pye3d_direction(th, ph)
                            gdx = gx * arrow_len
                            gdy = gy * arrow_len
                            tip = (int(round(cx2 + gdx)), int(round(cy2 + gdy)))
                            cv2.arrowedLine(rgb, center, tip,
                                            (0, 140, 255), 3, tipLength=0.3)

                    # ── projected eyeball sphere (yellow) ─────────────────────
                    ps = r3.get("projected_sphere")
                    if ps and model_ready:
                        sc   = ps["center"]
                        sax  = ps["axes"]
                        sang = float(ps["angle"])
                        scenter = (int(round(float(sc[0]))), int(round(float(sc[1]))))
                        sr0  = max(1, int(round(float(sax[0]) / 2)))
                        sr1  = max(1, int(round(float(sax[1]) / 2)))
                        if (scenter[0] - sr0 > -fw_src // 2
                                and scenter[0] + sr0 < fw_src * 3 // 2
                                and scenter[1] - sr1 > -fh_src // 2
                                and scenter[1] + sr1 < fh_src * 3 // 2):
                            cv2.ellipse(rgb, scenter, (sr0, sr1), sang,
                                        0, 360, (255, 220, 0), 1)

                    # ── info label ────────────────────────────────────────────
                    sphere  = r3.get("sphere", {})
                    ratio = tracker.accepted_ratio
                    source_label = {
                        "PRIMARY": "2D",
                        "ASSIST": "AUX",
                        "PREDICT": "PRED",
                        "LOST": "LOST",
                    }[decision.source]
                    source_label = tracker.visibility + " / " + source_label
                    if model_ready:
                        td = math.degrees(float(theta_v))
                        pd = math.degrees(float(phi_v))
                        sr = float(sphere.get("radius", 0))
                        self._det_info_vars[idx].set(
                            f"{source_label} raw={conf2:.2f} pass={ratio:.0%}  "
                            f"θ={td:+.1f}° φ={pd:+.1f}° eye_r={sr:.1f}mm m={conf3:.2f}")
                        if self._gaze_view:
                            gaze_confidence = conf3
                            if decision.source == "ASSIST":
                                gaze_confidence = min(gaze_confidence, 0.79)
                            elif decision.source == "PREDICT":
                                gaze_confidence = min(gaze_confidence, 0.35)
                            self._gaze_view.update(
                                idx, float(theta_v), float(phi_v), gaze_confidence)
                    else:
                        if tracker.first_primary_ts is None:
                            model_status = "waiting for trustworthy pupil"
                        elif warm_elapsed < 5.0 or tracker.primary_count < 60:
                            model_status = f"3D building {warm_elapsed:.1f}/5s {tracker.primary_count}/60 samples"
                        elif r3 and conf3 <= 0.11:
                            model_status = "3D INVALID: model out of physiological range"
                        else:
                            model_status = "3D unavailable"
                        self._det_info_vars[idx].set(
                            f"{source_label} raw={conf2:.2f} pass={ratio:.0%}  "
                            f"{model_status}; {decision.reason}")
                        if self._gaze_view:
                            self._gaze_view.invalidate(idx)

                except Exception as e:
                    self._export_sample(idx, frame_timestamp, "DETECTION_ERROR")
                    self._det_info_vars[idx].set(f"detect err: {e}")
                    self._trackers[idx].gaze_allowed = False
                    if self._gaze_view:
                        self._gaze_view.invalidate(idx)
            else:
                # Hough fallback
                blurred = cv2.GaussianBlur(frame, (7, 7), 1.5)
                circles = cv2.HoughCircles(
                    blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                    minDist=max(frame.shape) // 4,
                    param1=50, param2=25,
                    minRadius=8, maxRadius=max(frame.shape) // 3,
                )
                if circles is not None:
                    c = np.round(circles[0, 0]).astype(int)
                    cv2.circle(rgb, (c[0], c[1]), c[2], (0, 255, 0), 2)
                    cv2.circle(rgb, (c[0], c[1]), 2,    (0, 255, 0), -1)
        else:
            self._export_sample(idx, frame_timestamp, "DISABLED")
            self._det_info_vars[idx].set("")
            if self._gaze_view:
                self._gaze_view.invalidate(idx)

        if self.detect_vars[idx].get():
            self._debug_history[idx].append(frame.copy())
        if self._debug_visible():
            packet = self._build_debug_packet(
                idx, frame, algorithm_bgr,
                debug_result_2d, debug_result_3d,
            )
            self._debug_packets[idx] = packet
            self._debug_view.update_eye(idx, packet)

        # ── ROI rectangle overlay (cyan) ──────────────────────────────────────
        if roi is not None:
            rx, ry, rw, rh = roi
            rx = max(0, min(fw_src - 1, rx))
            ry = max(0, min(fh_src - 1, ry))
            rw = max(1, min(fw_src - rx, rw))
            rh = max(1, min(fh_src - ry, rh))
            cv2.rectangle(rgb, (rx, ry), (rx + rw, ry + rh), (0, 220, 255), 2)

        fh, fw = rgb.shape[:2]
        scale  = min(PREVIEW_W / fw, PREVIEW_H / fh)
        nw, nh = max(1, round(fw * scale)), max(1, round(fh * scale))
        pil    = Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BILINEAR)
        out    = Image.new("RGB", (PREVIEW_W, PREVIEW_H), (13, 17, 23))
        out.paste(pil, ((PREVIEW_W - nw) // 2, (PREVIEW_H - nh) // 2))

        photo = ImageTk.PhotoImage(image=out)
        self._photo[idx] = photo
        self._canvases[idx].delete("preview")
        self._canvases[idx].create_image(0, 0, anchor=tk.NW, image=photo, tags="preview")
        self._canvases[idx].tag_lower("preview")

    # ── close ─────────────────────────────────────────────────────────────────

    def _recording_metadata(self):
        return dict(app_version=APP_VERSION, eye_uids=list(self._uids), mode=self.mode_var.get(),
                    flip=[v.get() for v in self.flip_vars], roi=list(self._roi),
                    exposure=self.exp_val_var.get(), intensity=self.det_intensity_var.get(),
                    pupil_min=self.det_pmin_var.get(), pupil_max=self.det_pmax_var.get(),
                    detection_enabled=[v.get() for v in self.detect_vars],
                    coarse=self.det_coarse_var.get(), assist=self.assist_var.get(),
                    intrinsics=[dict(resolution=[c.width, c.height],
                        focal_length_px=_cam2_focal_length(c.width, c.height))
                        if c and c.width and c.height else None for c in self._captures])

    def _toggle_recording(self):
        if self._recorder:
            recorder, self._recorder = self._recorder, None
            try:
                path = recorder.close()
                self._record_status.set("Saved: " + path.name)
            except OSError as exc:
                self._record_status.set("EXPORT ERROR: " + str(exc)[:60])
                messagebox.showerror("Export error", str(exc))
        else:
            try:
                self._record_config = self._recording_metadata()
                self._recorder = SessionRecorder(DATA_ROOT / "recordings", self._record_config,
                                                 int(self._record_hz.get()))
                self._record_status.set("RECORDING: " + self._recorder.path.name)
            except (OSError, ValueError, tk.TclError) as exc:
                messagebox.showerror("Cannot start export", str(exc))

    def _recording_failed(self, exc):
        recorder, self._recorder = self._recorder, None
        if recorder:
            with contextlib.suppress(OSError):
                recorder.close()
        self._record_status.set("EXPORT FAILED - recording stopped")
        messagebox.showerror("Export stopped (disk error)", str(exc))

    def _export_sample(self, idx, timestamp, state, **values):
        if self._recorder:
            try:
                self._recorder.sample(idx, timestamp, self._prev_seq[idx], state, **values)
            except OSError as exc:
                self._recording_failed(exc)

    def _mark_recording(self):
        if self._recorder:
            try:
                self._recorder.event("user_marker", "button press")
            except OSError as exc:
                self._recording_failed(exc)

    def _on_close(self) -> None:
        if self._recorder:
            self._toggle_recording()
        self._save_settings()
        if self._gaze_view:
            self._gaze_view._alive = False
        if self._debug_view:
            self._debug_view.close()
        for cap in self._captures:
            if cap:
                cap.stop()
        for cap in self._captures:
            if cap:
                cap.join(timeout=2.0)
        self.root.destroy()


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    root = tk.Tk()
    PupilDualStreamApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
