@echo off
SETLOCAL
cd /d "%~dp0"

echo [INFO] 检查 Python 版本...
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] 未找到 Python，准备自动安装...
    goto :INSTALL_PYTHON
)
for /f "tokens=1,2" %%a in ('python --version 2^>^&1') do (
    echo %%b | findstr /r "^3\.11\." >nul
    if errorlevel 1 (
        echo [WARNING] 检测到 %%a %%b，版本不符，准备安装 Python 3.11.9...
        goto :INSTALL_PYTHON
    )
    echo [INFO] 检测到 %%a %%b
)

echo [INFO] 检查环境变量 DASHSCOPE_API_KEY...
if "%DASHSCOPE_API_KEY%"=="" (
    echo [ERROR] 环境变量 DASHSCOPE_API_KEY 未设置，请先配置。
    pause
    exit /b 1
)

echo [INFO] 检查端口 8000 是否被占用...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo [INFO] 发现端口占用，正在关闭旧进程 PID: %%a
    taskkill /F /PID %%a
)
REM 等待系统释放端口
timeout /t 2 /nobreak > nul

if not exist "myenv" (
    echo [INFO] 正在创建虚拟环境...
    python -m venv myenv
    echo [INFO] 正在安装依赖，请稍候...
    call myenv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call myenv\Scripts\activate.bat
)

echo [INFO] start_service...

python service.py
pause
exit /b 0

:INSTALL_PYTHON
set "PYTHON_EXE=python-3.11.9-amd64.exe"
set "PYTHON_URL=https://www.python.org/ftp/python/3.11.9/%PYTHON_EXE%"
echo [INFO] 正在从 %PYTHON_URL% 下载安装包...
curl -L -o "%temp%\%PYTHON_EXE%" %PYTHON_URL%
if %errorlevel% neq 0 (echo [ERROR] 下载失败，请检查网络连接。 & pause & exit /b 1)
echo [INFO] 正在启动静默安装 (可能需要管理员授权)...
start /wait "" "%temp%\%PYTHON_EXE%" /quiet PrependPath=1 Include_test=0
del "%temp%\%PYTHON_EXE%"
echo [SUCCESS] Python 3.11.9 安装尝试完成。
echo [重要] 请关闭此窗口并重新运行“一键启动.bat”以激活环境。
pause
exit /b 0