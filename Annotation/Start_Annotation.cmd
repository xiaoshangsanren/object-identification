@echo off
setlocal
call "%~dp0command\set_environment.cmd"
"%PACKAGE_ROOT%\runtime\python.exe" "%PACKAGE_ROOT%\command\launch_ui.py"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Service exited with code %EXIT_CODE%.
  pause
)
exit /b %EXIT_CODE%
