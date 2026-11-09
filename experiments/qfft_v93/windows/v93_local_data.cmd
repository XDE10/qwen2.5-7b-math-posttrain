@echo off
setlocal EnableExtensions
rem v93 local data build on the Windows workstation (CPU). Double-click, or run from any directory.
rem Runs the five steps of experiments/qfft_v93/README.md section 1 in order and stops at the first error.
rem Every step resumes or rebuilds deterministically, so running this file again is always safe.
rem Step 4 calls the DeepSeek API only for questions that still lack a usable trace.
rem Interpreter: set P1_PYTHON=C:\path\to\python.exe to override the default below.

pushd "%~dp0..\..\.." || goto :fail
set "PY=%P1_PYTHON%"
if not defined PY set "PY=D:\anaconda3\envs\genai-test\python.exe"
if not exist "%PY%" (
  echo Python not found: %PY%
  echo Set P1_PYTHON to the interpreter that has pandas, pyarrow and tokenizers.
  goto :fail
)
set "PYTHONUTF8=1"
echo Repository: %CD%
echo Python:     %PY%

call :step "1/5 benchmarks and sources at pinned revisions" -m experiments.qfft_v93.fetch_benchmarks --sources || goto :fail
call :step "2/5 1000-question plan (v93 replacement, decontamination)" -m experiments.qfft_v93.data_build plan || goto :fail
call :step "3/5 questions still without a usable DeepSeek trace" -m experiments.qfft_v93.data_build status || goto :fail
call :step "4/5 DeepSeek fill until 1000/1000 (API, resumable)" -m experiments.qfft_v93.teacher fill --fallback-official || goto :fail
call :step "5/5 build r1/, ds/, ds_verified/ and the teacher report" -m experiments.qfft_v93.data_build build --ds-verified || goto :fail

echo.
echo Done. Read data\qfft1000_v93\teacher_report.md, then upload the data with:
echo     experiments\qfft_v93\windows\v93_colab.cmd d
popd
pause
exit /b 0

:step
echo.
echo ===== %~1
"%PY%" %2 %3 %4 %5 %6 %7 %8 %9
exit /b %errorlevel%

:fail
echo.
echo FAILED - see the messages above. Fix the cause and run this file again; finished work is kept.
popd 2>nul
pause
exit /b 1
