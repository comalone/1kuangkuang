# 豆包API配置说明

## 环境变量配置

### 方式1: 在系统环境变量中配置

Windows系统:
```cmd
setx DOUBAO_API_KEY "your-doubao-api-key-here"
setx DOUBAO_API_URL "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
setx DOUBAO_MODEL "doubao-vision-pro-32k-2410128"
```

### 方式2: 在启动脚本中临时配置

修改 `一键启动.bat` 文件,在启动服务前添加:
```cmd
set DOUBAO_API_KEY=your-doubao-api-key-here
set DOUBAO_API_URL=https://ark.cn-beijing.volces.com/api/v3/chat/completions
set DOUBAO_MODEL=doubao-vision-pro-32k-2410128
```

### 方式3: 使用 .env 文件(推荐)

创建 `.env` 文件,内容如下:
```
# 通义千问API配置
DASHSCOPE_API_KEY=your-qwen-api-key

# 豆包API配置
DOUBAO_API_KEY=your-doubao-api-key
DOUBAO_API_URL=https://ark.cn-beijing.volces.com/api/v3/chat/completions
DOUBAO_MODEL=doubao-vision-pro-32k-2410128
```

然后安装 python-dotenv:
```bash
pip install python-dotenv
```

在 `service.py` 开头添加:
```python
from dotenv import load_dotenv
load_dotenv()
```

## API使用说明

### 1. 获取当前API提供商
```bash
GET /get_api_provider
```

响应示例:
```json
{
  "provider": "qwen",
  "qwen_configured": true,
  "doubao_configured": true
}
```

### 2. 切换API提供商
```bash
POST /set_api_provider
Content-Type: application/json

{
  "provider": "doubao"
}
```

响应示例:
```json
{
  "message": "API提供商已切换为 豆包",
  "provider": "doubao"
}
```

## 豆包API特点

1. **本地图片处理**: 豆包API使用base64编码直接发送图片,无需上传到OSS
2. **支持多图片**: 可以在一个请求中发送多张图片
3. **响应格式**: 与OpenAI兼容的响应格式

## 示例代码

### JavaScript前端调用示例
```javascript
// 获取当前API提供商
async function getCurrentProvider() {
  const response = await fetch('/get_api_provider');
  const data = await response.json();
  console.log('当前提供商:', data.provider);
  return data;
}

// 切换到豆包API
async function switchToDoubao() {
  const response = await fetch('/set_api_provider', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ provider: 'doubao' })
  });
  const data = await response.json();
  console.log(data.message);
}

// 切换到通义千问API
async function switchToQwen() {
  const response = await fetch('/set_api_provider', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ provider: 'qwen' })
  });
  const data = await response.json();
  console.log(data.message);
}
```

## 注意事项

1. 豆包API需要有效的API密钥才能使用
2. 图片会被转换为base64格式,较大的图片可能导致请求体积增大
3. 默认使用通义千问API,需要手动切换到豆包
4. 两个API可以同时配置,运行时动态切换
