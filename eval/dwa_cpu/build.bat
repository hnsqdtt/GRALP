@echo off
REM Build dwa_cpu.exe with MSVC. Locates a VS install via vswhere, sources
REM vcvars64, then compiles the single translation unit with /O2.
setlocal enabledelayedexpansion

set "HERE=%~dp0"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo [build] vswhere not found at "%VSWHERE%"
  exit /b 1
)

set "VSPATH="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -all -products * -property installationPath`) do (
  if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" set "VSPATH=%%i"
)
if "%VSPATH%"=="" (
  echo [build] no VS install with vcvars64.bat found
  exit /b 1
)

echo [build] using %VSPATH%
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
  echo [build] vcvars64 failed
  exit /b 1
)

cl /nologo /O2 /W3 /Fe:"%HERE%dwa_cpu.exe" /Fo:"%HERE%dwa_cpu.obj" "%HERE%dwa_cpu.c"
if errorlevel 1 (
  echo [build] compile failed
  exit /b 1
)

del "%HERE%dwa_cpu.obj" 2>nul
echo [build] OK: "%HERE%dwa_cpu.exe"
exit /b 0
