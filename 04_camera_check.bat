@echo off
setlocal
if not exist "%~dp0runtime\python.exe" (
  echo Run 01_install.bat first.
  pause
  exit /b 1
)
echo Close the tracker before testing. This opens BOTH cameras for up to 10 seconds.
"%~dp0runtime\python.exe" -I "%~dp0app\launch.py" --camera-check
pause
