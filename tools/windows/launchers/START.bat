@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0"
"python\python.exe" -X utf8 -u tools\windows\portable.py start %*
set "strata_result=%errorlevel%"
if not "%strata_result%"=="0" pause
popd
exit /b %strata_result%
