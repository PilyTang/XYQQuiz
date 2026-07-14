@echo off
setlocal
set "ROOT=%~dp0"
set "REPORT=%ROOT%diagnostics\self-test-latest"
if not exist "%REPORT%" mkdir "%REPORT%"
start "" /wait "%ROOT%XYQQuiz.exe" --self-test --report-dir "%REPORT%"
set "RESULT=%ERRORLEVEL%"
for /f "delims=" %%F in ('dir /b /a-d /o-d "%REPORT%\self-test-*.html" 2^>nul') do (
  start "" "%REPORT%\%%F"
  goto :finished
)
echo No HTML report was generated. Check the diagnostics folder.
:finished
if not "%RESULT%"=="0" echo Self-test failed with exit code %RESULT%.
pause
exit /b %RESULT%
