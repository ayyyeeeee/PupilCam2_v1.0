"""Read-only driver discovery plus real dual-stream acquisition diagnostic."""
import json
import pathlib
import sys
import time
from pupil_tracker import EyeCapture, _find_pupil_uids, APP_ROOT


def main():
    uids = _find_pupil_uids()
    captures = [EyeCapture(i, uid) for i, uid in enumerate(uids[:2])]
    report = dict(uids=uids, eyes=[], passed=False)
    try:
        for cap in captures:
            cap.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if len(captures) == 2 and all(c.get_seq() >= 30 for c in captures):
                break
            time.sleep(.05)
        for cap in captures:
            frame, seq, timestamp = cap.get_latest()
            report["eyes"].append(dict(uid=cap.uid, frames=seq, mode=cap.active_mode_label,
                shape=list(frame.shape) if frame is not None else None, error=cap.error,
                age_s=time.monotonic()-timestamp if timestamp else None))
        report["passed"] = len(captures) == 2 and all(
            e["frames"] >= 30 and e["shape"] == [400, 400] and not e["error"] and e["age_s"] < .5
            for e in report["eyes"])
    finally:
        for cap in captures:
            cap.stop()
        for cap in captures:
            cap.join(timeout=3)
    path = APP_ROOT / "data" / "camera_check.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["passed"] else 2)
