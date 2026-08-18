@echo off
setlocal
cd /d "%~dp0..\.."
runtime\python.exe -m PAD_Lite b0 --config PAD_Lite\configs\default.json --fold all %*
