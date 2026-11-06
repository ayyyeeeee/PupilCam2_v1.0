@echo off
setlocal
if not exist "%~dp0runtime\python.exe" (
  echo Run 01_install.bat first.
  pause
  exit /b 1
)
"%~dp0runtime\python.exe" -I "%~dp0app\launch.py"
if errorlevel 1 pause
