# 豆包API集成更新说明

## 更新内容

本次更新为"一框框系统"添加了豆包(Doubao)视觉API的支持,现在您可以在通义千问和豆包两个AI模型之间自由切换进行图片理解分析。

## 主要功能

### 1. 双API支持
- **通义千问 (Qwen-VL)**: 阿里云的多模态大模型
- **豆包 (Doubao Vision)**: 字节跳动的视觉理解模型

### 2. 动态切换
- 在截图结果页面右上角可以实时切换API提供商
- 切换后立即生效,无需重启服务
- 自动检测API配置状态,未配置的API按钮会被禁用

### 3. 智能提示
- 显示当前使用的API提供商
- 切换成功后有视觉反馈
- 配置错误时有明确的错误提示

## 配置方法

### 方式1: 环境变量配置(推荐)

在Windows系统中设置环境变量:

```cmd
# 通义千问API配置
setx DASHSCOPE_API_KEY "your-qwen-api-key"

# 豆包API配置
setx DOUBAO_API_KEY "your-doubao-api-key"
setx DOUBAO_API_URL "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
setx DOUBAO_MODEL "doubao-vision-pro-32k-2410128"
```

**注意**: 设置环境变量后需要重新打开命令行窗口才能生效。

### 方式2: 修改启动脚本

编辑 `一键启动.bat` 文件,在启动服务前添加:

```batch
@echo off
chcp 65001 >nul

REM 设置API密钥
set DASHSCOPE_API_KEY=your-qwen-api-key
set DOUBAO_API_KEY=your-doubao-api-key
set DOUBAO_API_URL=https://ark.cn-beijing.volces.com/api/v3/chat/completions
set DOUBAO_MODEL=doubao-vision-pro-32k-2410128

REM 启动服务
cd /d "%~dp0"
call myenv\Scripts\activate.bat
python service.py
```

### 方式3: 使用.env文件

1. 安装 python-dotenv:
```bash
pip install python-dotenv
```

2. 在项目根目录创建 `.env` 文件:
```
DASHSCOPE_API_KEY=your-qwen-api-key
DOUBAO_API_KEY=your-doubao-api-key
DOUBAO_API_URL=https://ark.cn-beijing.volces.com/api/v3/chat/completions
DOUBAO_MODEL=doubao-vision-pro-32k-2410128
```

3. 在 `service.py` 开头添加:
```python
from dotenv import load_dotenv
load_dotenv()
```

## 使用说明

### 1. 启动服务

运行 `一键启动.bat` 启动服务。启动时会显示API配置状态:

```
警告: 环境变量 DASHSCOPE_API_KEY 未配置,通义千问 AI 分析功能将不可用
警告: 环境变量 DOUBAO_API_KEY 未配置,豆包 AI 分析功能将不可用
```

如果两个API都配置成功,则不会显示警告。

### 2. 切换API

在截图结果页面(`screenshot_result.html`)的右上角:
- 显示当前使用的API提供商
- 点击"通义千问"或"豆包"按钮切换API
- 未配置的API按钮会显示为灰色且不可点击

### 3. 查看分析结果

- 截图后,系统会自动使用当前选择的API进行图片分析
- 分析结果会以打字机效果显示在页面中央
- 不同的API可能给出不同的分析结果

## API对比

| 特性 | 通义千问 | 豆包 |
|------|---------|------|
| 图片上传方式 | OSS云存储 | Base64编码 |
| 响应速度 | 较快 | 较快 |
| 多图支持 | 支持 | 支持 |
| 中文理解 | 优秀 | 优秀 |
| 详细程度 | 详细 | 详细 |

## API接口说明

### 获取当前API提供商
```
GET /get_api_provider
```

响应:
```json
{
  "provider": "qwen",
  "qwen_configured": true,
  "doubao_configured": true
}
```

### 切换API提供商
```
POST /set_api_provider
Content-Type: application/json

{
  "provider": "doubao"
}
```

响应:
```json
{
  "message": "API提供商已切换为 豆包",
  "provider": "doubao"
}
```

## 技术细节

### 通义千问工作流程
1. 获取上传凭证
2. 上传图片到OSS
3. 调用Qwen-VL API
4. 返回分析结果

### 豆包工作流程
1. 读取本地图片
2. 转换为Base64编码
3. 直接调用Doubao API
4. 返回分析结果

## 常见问题

### Q: 为什么切换API后没有反应?
A: 请确保对应的API密钥已正确配置,并且网络连接正常。

### Q: 可以同时使用两个API吗?
A: 可以同时配置两个API,但每次分析只会使用当前选择的一个API。

### Q: 豆包API的密钥在哪里获取?
A: 请访问火山引擎官网注册并申请豆包API密钥。

### Q: 图片太大会影响性能吗?
A: 豆包使用Base64编码,较大的图片会增加请求体积。建议图片大小控制在2MB以内。

## 更新日志

### v1.2.0 (2026-01-04)
- ✨ 新增豆包视觉API支持
- ✨ 新增API提供商动态切换功能
- ✨ 新增API配置状态检测
- 🎨 优化截图结果页面UI
- 📝 添加详细的配置文档

## 技术支持

如有问题,请访问项目主页或联系作者:
- 抖音: [@禾禾椰](https://www.douyin.com/user/MS4wLjABAAAAOyqvejBV5f3GmXNbmeCmkCRQJ84Lcluy1uMeWwKa7o0)
- 项目版本: v1.2.0
- 版权所有: 一框框系统 © 2026
