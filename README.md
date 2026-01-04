# 一框框系统 (1kuangkuang)

**手势驱动的 AI 视觉助手 | Gesture-Driven AI Vision Assistant**

这是一个基于 **MediaPipe** 和 **FastAPI** 开发的智能交互系统。它允许用户通过自然手势在视频流中框选区域，并自动调用 **阿里云通义千问 (Qwen-VL)** 或 **字节跳动豆包 (Doubao Vision)** 大模型对截图内容进行深度分析。

系统采用单页应用 (SPA) 架构，提供无缝的交互体验，支持实时手势识别、智能截图、AI 分析和语音/视觉反馈。

## ✨ 核心功能

-   **👆 手势框选截图**:
    -   **指指点点**: 食指作为指针触发检测。
    -   **静止锁定**: 悬停 1 秒锁定起始点。
    -   **拉框截图**: 移动手指绘制矩形，再次悬停 1 秒完成截图。
-   **🧠 双 AI 引擎切换**:
    -   内置 **通义千问 (Qwen-VL)** 和 **豆包 (Doubao Vision)** 支持。
    -   支持在界面设置中实时切换 AI 模型和版本。
-   **⚡ 单页无缝体验**:
    -   拍摄与分析界面平滑过渡，无需页面跳转。
    -   **OK 手势**: 在分析结果页比划 "OK" (食指拇指捏合) 即可一键返回拍摄模式。
-   **⚙️ 实时配置**:
    -   支持画面旋转 (90°/180°/270°) 和镜像翻转。
    -   内置多种 AI 角色提示词 (解题、润色、创作等)。
-   **📹 灵活采集**: 支持 USB 摄像头及 ESP32-CAM 等网络流。

## �️ 技术栈

-   **后端**: Python 3.8+, FastAPI, Uvicorn
-   **视觉处理**: OpenCV, MediaPipe (Lite 模型优化), NumPy
-   **AI 服务**: 
    -   阿里云 DashScope SDK (Qwen-VL)
    -   字节跳动豆包视觉 API (Doubao Vision)
-   **前端**: HTML5, CSS3, JavaScript (原生 SPA 架构, WebSocket 通信)

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install fastapi uvicorn opencv-python mediapipe numpy requests dashscope python-dotenv
```

### 2. 配置 API Key
在环境变量中配置您的 AI API 密钥：

**通义千问** (阿里云):
```bash
set DASHSCOPE_API_KEY=your-qwen-api-key
```

**豆包** (字节跳动):
```bash
set DOUBAO_API_KEY=your-doubao-api-key
set DOUBAO_API_URL=https://ark.cn-beijing.volces.com/api/v3/chat/completions
set DOUBAO_MODEL=doubao-vision-pro-32k-2410128
```

### 3. 运行服务
```bash
python service.py
```
访问地址：`http://localhost:8000`

## 🎮 操作指南

| 动作 | 触发效果 | 视觉反馈 |
| :--- | :--- | :--- |
| **伸出食指** | 进入检测模式 | 状态切换为 `DETECTING` |
| **保持静止 1s** | 锁定起始点 | 出现绿色矩形框 |
| **移动食指** | 调整截图区域 | 矩形框随手指实时拉伸 |
| **再次静止 1s** | 执行截图并上传 | 绿框消失，自动跳转结果页 |
| **比划 "OK"** | (结果页) 返回拍摄 | 自动重置状态机并跳转回主页 |

## 🎨 AI 分析模式

系统内置了多种 AI 提示词模式，定义在 `prompts.json` 中：

-   **💡 解题思路**: 识别题目并提供步骤清晰的解题指导。
-   **🧠 深度理解**: 解析图片背后的逻辑、背景及情感含义。
-   **✨ 文章润色**: 识别文字并进行专业级润色优化。
-   **🔥 爆文创作**: 基于图片生成小红书/微博风格的爆款文案。
-   **🛡️ 内容审核**: 检查图片内容的合规性与逻辑错误。
-   **🎓 论文精修**: 优化学术论文片段的表达与逻辑。
-   **🎭 剧本扩写**: 根据画面扩写电影级剧本情节。

## 📂 项目结构

```text
├── service.py                # FastAPI 后端逻辑、手势识别与 AI 调用
├── index.html           # 主拍摄界面
├── screenshot_result.html    # 结果展示与 AI 分析界面
├── prompts.json              # AI 提示词配置文件
├── test_doubao_api.py        # 豆包 API 测试脚本
├── 豆包API使用说明.md      # 豆包 API 配置文档
├── uploads/                  # 截图文件存储目录
└── README.md                 # 项目说明文档
```


## 📄 许可证
本项目基于 Apache v2 许可证开源。
