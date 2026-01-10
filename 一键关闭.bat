@echo off
chcp 65001 >nul
SETLOCAL
echo [INFO] 正在停止运行在 8000 端口的服务...

for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo [INFO] 正在关闭进程 PID: %%a
    taskkill /F /PID %%a
)

echo [INFO] 服务已停止。
pause