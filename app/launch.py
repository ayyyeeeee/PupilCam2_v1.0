"""Isolated entry point; only this bundle's modules and runtime are used."""
import os
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent.parent
os.environ["TCL_LIBRARY"] = str(root / "runtime" / "tcl" / "tcl8.6")
os.environ["TK_LIBRARY"] = str(root / "runtime" / "tcl" / "tk8.6")
sys.path.insert(0, str(root / "app"))
if "--selfcheck" in sys.argv:
    import selfcheck
    selfcheck.main()
elif "--camera-check" in sys.argv:
    import camera_check
    camera_check.main()
else:
    import pupil_tracker
    pupil_tracker.main()
