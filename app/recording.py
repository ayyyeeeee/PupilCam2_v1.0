"""Small, explicit measurements: no video, no interpolation, no fake gaze.

All calls originate on the Tk thread. Rate limiting affects sample rows only;
state transitions and setting changes are always kept in events.csv.
"""
import collections
import csv
import datetime
import json
import math
import pathlib
import time
import uuid


class SessionRecorder:
    FIELDS = ("t_s", "eye", "frame_id", "state", "pupil_valid", "gaze_valid",
              "confidence_2d", "boundary_support", "model_confidence",
              "cx_px", "cy_px", "axis0_px", "axis1_px", "angle_deg",
              "gaze_x", "gaze_y", "gaze_z")

    def __init__(self, root, metadata, hz=20, clock=time.monotonic):
        if not 1 <= hz <= 60:
            raise ValueError("sample rate must be between 1 and 60 Hz per eye")
        self.clock, self.hz = clock, hz
        self.start = clock()
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = pathlib.Path(root) / (stamp + "_" + uuid.uuid4().hex[:6])
        self.path.mkdir(parents=True, exist_ok=False)
        self.last = {}
        self.states = {}
        self.counts = collections.Counter()
        self.closed = False
        self.flushed = self.start
        meta = dict(metadata, schema_version=1, sample_rate_limit_per_eye_hz=hz,
                    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    timestamp_origin="host monotonic frame receipt; NOT hardware synchronized",
                    coordinates="post-flip camera pixels: x right, y down; ellipse axes are diameters",
                    gaze_coordinates="post-flip eye-camera 3D unit vector, NOT screen point",
                    screen_calibrated=False, invalid_values="empty, never zero-filled or interpolated",
                    sampling="latest processed frames, rate limited; no resampling or averaging")
        (self.path / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self.samples = (self.path / "samples.csv").open("w", newline="", encoding="utf-8-sig")
        self.events_file = (self.path / "events.csv").open("w", newline="", encoding="utf-8-sig")
        self.writer = csv.DictWriter(self.samples, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.events = csv.writer(self.events_file)
        self.events.writerow(("t_s", "eye", "event", "detail"))

    def event(self, name, detail, eye="", timestamp=None):
        if not self.closed:
            ts = self.clock() if timestamp is None else timestamp
            self.events.writerow((round(max(0, ts-self.start), 6), eye, name, detail))

    @staticmethod
    def number(value):
        try:
            value = float(value)
            return round(value, 5) if math.isfinite(value) else ""
        except (TypeError, ValueError):
            return ""

    def sample(self, eye, timestamp, frame_id, state, ellipse=None, gaze=None,
               confidence=0, boundary=0, model_confidence=0):
        if self.closed or not math.isfinite(timestamp) or timestamp < self.start:
            return
        if timestamp < self.last.get(eye, -math.inf):
            return
        pupil_ok = ellipse is not None and state == "VISIBLE"
        if pupil_ok:
            values = (*ellipse["center"], *ellipse["axes"], ellipse["angle"])
            pupil_ok = all(self.number(v) != "" for v in values) and min(ellipse["axes"]) > 0
        gaze_ok = pupil_ok and gaze is not None and len(gaze) == 3 and all(self.number(v) != "" for v in gaze)
        transition = (state, bool(pupil_ok), bool(gaze_ok))
        if transition != self.states.get(eye):
            self.event("validity", json.dumps(transition), eye, timestamp)
            self.states[eye] = transition
        if timestamp-self.last.get(eye, -math.inf) + 1e-9 < 1/self.hz:
            return
        self.last[eye] = timestamp
        row = dict(t_s=round(timestamp-self.start, 6), eye=eye, frame_id=frame_id,
                   state=state, pupil_valid=int(pupil_ok), gaze_valid=int(gaze_ok),
                   confidence_2d=self.number(confidence), boundary_support=self.number(boundary),
                   model_confidence=self.number(model_confidence))
        if pupil_ok:
            row.update(zip(("cx_px", "cy_px", "axis0_px", "axis1_px", "angle_deg"),
                           map(self.number, values)))
        if gaze_ok:
            row.update(zip(("gaze_x", "gaze_y", "gaze_z"), map(self.number, gaze)))
        self.writer.writerow(row)
        self.counts[f"eye{eye}_rows"] += 1
        self.counts[f"eye{eye}_pupil_valid"] += int(pupil_ok)
        self.counts[f"eye{eye}_gaze_valid"] += int(gaze_ok)
        if self.clock()-self.flushed >= 1:
            self.samples.flush()
            self.events_file.flush()
            self.flushed = self.clock()

    def close(self):
        if self.closed:
            return self.path
        try:
            self.event("stop", "normal stop")
        finally:
            self.closed = True
            self.samples.close()
            self.events_file.close()
        summary = dict(duration_s=round(self.clock()-self.start, 3), counts=dict(self.counts),
                       note="Valid sample fraction is availability, NOT detection accuracy. No ground truth.")
        (self.path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return self.path
