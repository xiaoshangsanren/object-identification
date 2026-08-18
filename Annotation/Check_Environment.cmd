@echo off
setlocal
call "%~dp0command\set_environment.cmd"
"%PACKAGE_ROOT%\runtime\python.exe" "%PACKAGE_ROOT%\command\check_environment.py"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %EXIT_CODE%
