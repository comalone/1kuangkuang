@echo off
chcp 65001 >nul
echo ========================================
echo 豆包API配置助手
echo ========================================
echo.

echo 此脚本将帮助您配置豆包API环境变量
echo.

:input_api_key
set /p DOUBAO_KEY="请输入您的豆包API密钥: "
if "%DOUBAO_KEY%"=="" (
    echo 错误: API密钥不能为空!
    goto input_api_key
)

echo.
echo 是否使用默认的API配置?
echo   API地址: https://ark.cn-beijing.volces.com/api/v3/chat/completions
echo   模型名称: doubao-vision-pro-32k-2410128
echo.
set /p USE_DEFAULT="使用默认配置? (Y/N, 默认Y): "

if /i "%USE_DEFAULT%"=="N" (
    set /p DOUBAO_URL="请输入API地址: "
    set /p DOUBAO_MDL="请输入模型名称: "
) else (
    set DOUBAO_URL=https://ark.cn-beijing.volces.com/api/v3/chat/completions
    set DOUBAO_MDL=doubao-vision-pro-32k-2410128
)

echo.
echo ========================================
echo 配置信息确认:
echo ========================================
echo API密钥: %DOUBAO_KEY:~0,10%...
echo API地址: %DOUBAO_URL%
echo 模型名称: %DOUBAO_MDL%
echo ========================================
echo.

set /p CONFIRM="确认配置? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo 配置已取消
    pause
    exit /b
)

echo.
echo 正在设置环境变量...

setx DOUBAO_API_KEY "%DOUBAO_KEY%" >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 设置 DOUBAO_API_KEY 失败
    pause
    exit /b 1
)

setx DOUBAO_API_URL "%DOUBAO_URL%" >nul 2>&1
if %errorlevel% neq 0 (
    echo 警告: 设置 DOUBAO_API_URL 失败,将使用默认值
)

setx DOUBAO_MODEL "%DOUBAO_MDL%" >nul 2>&1
if %errorlevel% neq 0 (
    echo 警告: 设置 DOUBAO_MODEL 失败,将使用默认值
)

echo.
echo ========================================
echo ✓ 配置完成!
echo ========================================
echo.
echo 环境变量已设置:
echo   DOUBAO_API_KEY = %DOUBAO_KEY:~0,10%...
echo   DOUBAO_API_URL = %DOUBAO_URL%
echo   DOUBAO_MODEL = %DOUBAO_MDL%
echo.
echo 注意事项:
echo 1. 请重新打开命令行窗口以使环境变量生效
echo 2. 如果使用一键启动.bat,请重新运行该脚本
echo 3. 可以在截图结果页面右上角切换API提供商
echo.

set /p TEST_API="是否立即测试豆包API? (Y/N): "
if /i "%TEST_API%"=="Y" (
    echo.
    echo 正在测试豆包API...
    echo 请确保 uploads 目录中有测试图片
    echo.
    
    REM 临时设置环境变量用于测试
    set DOUBAO_API_KEY=%DOUBAO_KEY%
    set DOUBAO_API_URL=%DOUBAO_URL%
    set DOUBAO_MODEL=%DOUBAO_MDL%
    
    if exist "myenv\Scripts\activate.bat" (
        call myenv\Scripts\activate.bat
        python test_doubao_api.py
    ) else (
        echo 错误: 虚拟环境不存在,请先运行 一键启动.bat
    )
)

echo.
pause
