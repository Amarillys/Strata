@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0"
"python\python.exe" -X utf8 -u tools\windows\portable.py configure %*
set "strata_result=%errorlevel%"
echo.
pause
popd
exit /b %strata_result%
