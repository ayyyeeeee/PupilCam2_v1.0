"""Offline invariants, not a claim of real-world accuracy."""
import csv
import importlib.metadata
import json
import math
import pathlib
import tempfile
import unittest
import cv2
import numpy as np
import pupil_tracker as app
from recording import SessionRecorder


def close_test_root(root):
    root.update_idletasks()
    for callback in root.tk.splitlist(root.tk.call('after', 'info')):
        # Cancel timers without deleting commands owned by child widgets;
        # widget.destroy() must delete those commands exactly once.
        root.tk.call('after', 'cancel', callback)
    root.destroy()


class Checks(unittest.TestCase):
    def pupil(self, axes=(45, 65)):
        gray = np.full((400, 400), 150, np.uint8)
        ellipse = dict(center=(200., 200.), axes=axes, angle=25.)
        cv2.ellipse(gray, ((200., 200.), axes, 25.), 15, -1)
        return gray, dict(ellipse=ellipse, confidence=.99)

    def test_occlusion_and_recovery(self):
        t = app.TemporalPupilAssist()
        gray, r = self.pupil()
        for i in range(3):
            d = t.process(r, gray, None, 10, 150, 1+i/30, True)
            self.assertEqual(d.primary_accepted, i == 2)
        blank = np.full_like(gray, 150)
        d = t.process(r, blank, None, 10, 150, 1.1, True)
        self.assertFalse(d.primary_accepted)
        self.assertFalse(t.gaze_allowed)
        for i in range(3):
            d = t.process(r, gray, None, 10, 150, 1.2+i/30, True)
            self.assertEqual(d.primary_accepted, i == 2)
        d = t.process(r, gray, None, 10, 150, 2., True)
        self.assertFalse(d.primary_accepted, "stale gap requires recovery")

    def test_oblique_pupil(self):
        t = app.TemporalPupilAssist()
        gray, r = self.pupil()
        for i in range(6):
            t.process(r, gray, None, 10, 150, 1+i/30, False)
        gray, r = self.pupil((18, 65))
        for i in range(3):
            d = t.process(r, gray, None, 10, 150, 1.2+i/30, False)
        self.assertTrue(d.primary_accepted, d.reason)

    def test_no_seed_recovery(self):
        t = app.TemporalPupilAssist()
        gray, r = self.pupil()
        class FakeDetector:
            def update_properties(self, props): pass
            def detect(self, frame, roi): return r
        t._local_detector = FakeDetector()
        for _ in range(3):
            result = t.refine_primary(dict(confidence=0), gray, None, 10, 150)
        self.assertGreater(result["confidence"], .9)

    def test_export(self):
        now = [100.]
        with tempfile.TemporaryDirectory() as tmp:
            logger = SessionRecorder(tmp, dict(test=True), 20, clock=lambda: now[0])
            _, r = self.pupil()
            logger.sample(0, 100, 1, "VISIBLE", r["ellipse"], (0, 0, -1), .99, 1, .9)
            logger.sample(0, 100.01, 2, "OCCLUDED/UNCERTAIN", r["ellipse"], (0, 0, -1))
            logger.sample(0, 100.06, 3, "OCCLUDED/UNCERTAIN", r["ellipse"], (0, 0, -1))
            now[0] = 101
            logger.close()
            logger.close()
            with (logger.path / "samples.csv").open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["pupil_valid"], "1")
            self.assertEqual(rows[1]["pupil_valid"], "0")
            self.assertEqual(rows[1]["cx_px"], "")
            self.assertEqual(rows[1]["gaze_x"], "")
            self.assertIn("0.01", (logger.path / "events.csv").read_text(encoding="utf-8-sig"))

    def test_dependencies(self):
        requirements = (app.APP_ROOT / "requirements.txt").read_text().splitlines()
        for line in requirements:
            name, version = line.split("==")
            self.assertEqual(importlib.metadata.version(name), version)
        self.assertAlmostEqual(np.linalg.norm(app._pye3d_direction(1., 2.)), 1.)

    def test_render_export_integration(self):
        from unittest.mock import patch
        import time
        root = app.tk.Tk()
        root.withdraw()
        gray, r = self.pupil()
        class Fake2D:
            def detect(self, *args, **kwargs): return dict(r)
        class Fake3D:
            def update_and_detect(self, *args):
                return dict(theta=1., phi=2., confidence=.99, model_confidence=.99)
        try:
            with tempfile.TemporaryDirectory() as tmp, \
                 patch.object(app, 'DATA_ROOT', pathlib.Path(tmp)), \
                 patch.object(app.PupilDualStreamApp, '_open_cameras'), \
                 patch.object(app.PupilDualStreamApp, '_tick'), \
                 patch.object(app.PupilDualStreamApp, '_load_settings'):
                ui = app.PupilDualStreamApp(root)
                ui._detectors[0] = Fake2D()
                ui._det3d[0] = Fake3D()
                ui.flip_vars[0].set(False)
                ui._toggle_recording()
                path = ui._recorder.path
                start = time.monotonic()
                for i in range(10):
                    ui._prev_seq[0] = i+1
                    ui._render(0, gray, start+i*.1)
                ui._render(0, np.full_like(gray, 150), start+1.1)
                ui._toggle_recording()
                with (path/'samples.csv').open(encoding='utf-8-sig',newline='') as f:
                    rows = list(csv.DictReader(f))
                self.assertTrue(any(row['pupil_valid'] == '1' for row in rows))
                self.assertTrue(all(row['gaze_valid'] == '0' for row in rows), '3D warmup must not export gaze')
                self.assertEqual(rows[-1]['pupil_valid'], '0')
                self.assertEqual(rows[-1]['cx_px'], '')
        finally:
            close_test_root(root)

    def test_ui_layout_and_no_invalid_3d(self):
        from unittest.mock import patch
        root = app.tk.Tk()
        root.withdraw()
        try:
            with patch.object(app.PupilDualStreamApp, "_open_cameras"), \
                 patch.object(app.PupilDualStreamApp, "_tick"), \
                 patch.object(app.PupilDualStreamApp, "_load_settings"):
                ui = app.PupilDualStreamApp(root)
            root.update_idletasks()
            original = [(c.winfo_reqwidth(), c.winfo_reqheight()) for c in ui._canvases]
            root.geometry("1460x960")
            root.deiconify()
            root.update()
            positions = [(c.winfo_rootx(), c.winfo_rooty(), c.winfo_width(), c.winfo_height()) for c in ui._canvases]
            for text in ("ok", "uncertain gaze "*80, ""):
                for var in ui._det_info_vars:
                    var.set(text)
                root.update_idletasks()
                self.assertEqual(original, [(c.winfo_reqwidth(), c.winfo_reqheight()) for c in ui._canvases])
                self.assertEqual(positions, [(c.winfo_rootx(), c.winfo_rooty(), c.winfo_width(), c.winfo_height()) for c in ui._canvases])
            class Reject3D:
                def update_and_detect(self, *args):
                    raise AssertionError("invalid pupil reached 3D")
            ui.detect_vars[0].set(True)
            ui._det3d[0] = Reject3D()
            for i in range(5):
                ui._render(0, np.full((400, 400), 140, np.uint8), 1+i/30)
            self.assertNotIn("detect err", ui._det_info_vars[0].get())
            self.assertEqual(len(ui._canvases[0].find_withtag("preview")), 1)
        finally:
            close_test_root(root)


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
