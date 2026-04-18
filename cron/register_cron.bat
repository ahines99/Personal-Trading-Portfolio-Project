@echo off
REM ============================================================================
REM register_cron.bat
REM ----------------------------------------------------------------------------
REM Register the Tradier daily poll as a Windows Scheduled Task.
REM Runs daily at 16:30 (4:30 PM) local time. Skips weekends internally.
REM
REM Run this ONCE from an elevated cmd (Run as Administrator) — schtasks
REM needs admin rights to install scheduled tasks for the current user.
REM
REM To verify: schtasks /Query /TN TradierDailyPoll
REM To remove: schtasks /Delete /TN TradierDailyPoll /F
REM To run on demand: schtasks /Run /TN TradierDailyPoll
REM ============================================================================

setlocal

set "SCRIPT=%~dp0poll_tradier_daily.bat"

echo.
echo Registering Windows Scheduled Task "TradierDailyPoll"
echo Wrapper script: %SCRIPT%
echo Schedule: Daily at 16:30 local time
echo.

schtasks /Create ^
    /SC DAILY ^
    /TN "TradierDailyPoll" ^
    /TR "\"%SCRIPT%\"" ^
    /ST 16:30 ^
    /RL LIMITED ^
    /F

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS — task registered.
    echo.
    echo To run on demand:        schtasks /Run /TN TradierDailyPoll
    echo To check next run time:  schtasks /Query /TN TradierDailyPoll /V /FO LIST ^| findstr "Next Run Time"
    echo To remove the task:      schtasks /Delete /TN TradierDailyPoll /F
) else (
    echo.
    echo FAILED — schtasks returned %ERRORLEVEL%.
    echo Try running this script as Administrator.
)

endlocal
