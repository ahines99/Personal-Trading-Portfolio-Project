@echo off
REM ============================================================================
REM poll_tradier_daily.bat
REM ----------------------------------------------------------------------------
REM Daily Tradier options polling job (Windows Task Scheduler entry point).
REM Polls top 1500 tickers, runs each through the SMV adapter, writes one
REM parquet snapshot per day to data/cache/options/tradier_daily/<YYYY-MM-DD>.parquet
REM
REM Schedule: 16:30 ET on US trading days (after options settle at 16:15 ET).
REM Runtime: ~4-5 minutes for 1500 tickers (~3 expirations each, ThreadPool=8).
REM API budget: ~6000 Tradier calls/run, well under free-tier limits.
REM
REM Skips weekends — early-exit if Saturday/Sunday.
REM Logs to logs/tradier_daily/<YYYY-MM-DD>.log
REM
REM Setup (one-time):
REM   schtasks /Create /SC DAILY /TN "TradierDailyPoll" ^
REM     /TR "C:\Users\Alex Hines\OneDrive\Documents\Personal Projects\Personal-Trading-Portfolio-Project\cron\poll_tradier_daily.bat" ^
REM     /ST 16:30 /F
REM
REM Or run register_cron.bat in this directory.
REM ============================================================================

setlocal

REM Skip weekends — markets closed
for /f "skip=1" %%d in ('wmic path win32_localtime get dayofweek') do (
    if not "%%d"=="" (
        if "%%d"=="0" goto :skip_weekend
        if "%%d"=="6" goto :skip_weekend
    )
)

REM Date for log file (YYYY-MM-DD)
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value') do set datetime=%%i
set today=%datetime:~0,4%-%datetime:~4,2%-%datetime:~6,2%

REM Resolve project root (parent of this script's dir)
set "PROJ=%~dp0.."
pushd "%PROJ%"

REM Ensure log dir exists
if not exist "logs\tradier_daily" mkdir "logs\tradier_daily"

REM Run the poll. Output to dated log file.
echo [%date% %time%] Starting Tradier daily poll for %today%
python run_options_setup.py --daily-poll --max-tickers 1500 > "logs\tradier_daily\%today%.log" 2>&1
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% NEQ 0 (
    echo [%date% %time%] FAILED with exit code %EXITCODE% — see logs\tradier_daily\%today%.log
) else (
    echo [%date% %time%] Completed successfully
)

popd
endlocal
exit /b %EXITCODE%

:skip_weekend
echo [%date% %time%] Weekend — skipping poll
exit /b 0
