@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0copy_to_usb.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" echo Copy failed with exit code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
