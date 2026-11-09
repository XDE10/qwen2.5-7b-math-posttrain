@echo off
setlocal EnableExtensions
rem Runs the v93 Colab pipeline from Windows through WSL2 (the Colab CLI does not run on native Windows).
rem   v93_colab.cmd setup                       one-time: uv + google-colab-cli + Colab sign-in inside WSL
rem   v93_colab.cmd ping                        one-minute end-to-end check on a CPU VM
rem   v93_colab.cmd PHASE [options]             PHASE is 0, 1, 2, d, 3, 4, 5, 6 or 7 of pipeline.sh,
rem                                             e.g.  v93_colab.cmd 3 --gpu G4 --wait-min 120
rem   v93_colab.cmd status                      stage markers, training progress, evaluation completeness
rem   v93_colab.cmd attach SESSION              follow a detached job again
rem Double-clicking without arguments asks for one of these.

pushd "%~dp0..\..\.." || goto :end
set "REPO=%CD%"
popd
where wsl.exe >nul 2>nul
if errorlevel 1 (
  echo WSL is not installed. In an administrator PowerShell run:  wsl --install
  echo then reboot and run:  %~nx0 setup
  goto :end
)

set "ARGS=%*"
if not "%~1"=="" goto :run
echo Phases:    0 benchmarks+models   1 smoke   2 anchors   d upload data   3 DS pair
echo            4 GRPO   5 R1 pair   6 GRPO from SFT   7 score+figures
echo Shortcuts: setup   ping   status   attach SESSION
set /p "ARGS=Phase or shortcut: "

:run
for /f "tokens=1,*" %%a in ("%ARGS%") do (
  set "FIRST=%%a"
  set "REST=%%b"
)
set "SCRIPT=bash experiments/qfft_v93/colab/pipeline.sh %ARGS%"
if /i "%FIRST%"=="setup" set "SCRIPT=bash experiments/qfft_v93/colab/setup_wsl.sh"
if /i "%FIRST%"=="ping" set "SCRIPT=bash experiments/qfft_v93/colab/launch.sh --gpu cpu ping:30 %REST%"
if /i "%FIRST%"=="status" set "SCRIPT=bash experiments/qfft_v93/colab/launch.sh --gpu cpu status %REST%"
if /i "%FIRST%"=="attach" set "SCRIPT=bash experiments/qfft_v93/colab/launch.sh --attach %REST%"

echo Repository: %REPO%
echo WSL:        %SCRIPT%
rem -e runs bash directly (no extra shell layer); -l loads ~/.profile so ~/.local/bin (uv tools) is on PATH.
rem Scripts are started with "bash script.sh" because files on /mnt/d may lack the executable bit.
wsl.exe --cd "%REPO%" -e bash -lc "%SCRIPT%"
set "CODE=%errorlevel%"
echo.
if "%CODE%"=="0" (echo Finished.) else if "%CODE%"=="75" (
  echo Detached: the job keeps running on the VM and releases it when done. Follow it with: %~nx0 attach SESSION
) else (echo Exit code %CODE%. Re-running the same command resumes from Google Drive.)

:end
pause
