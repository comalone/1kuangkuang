"""
豆包API测试脚本
用于测试豆包视觉API的集成是否正常工作
"""

import os
import requests
import base64
from pathlib import Path

# 配置
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "your-api-key-here")
DOUBAO_API_URL = os.getenv("DOUBAO_API_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL", "doubao-vision-pro-32k-2410128")

def convert_image_to_base64(image_path):
    """将图片转换为base64编码"""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    ext = Path(image_path).suffix.lower()
    mime_type = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
    
    base64_data = base64.b64encode(image_data).decode('utf-8')
    data_url = f"data:{mime_type};base64,{base64_data}"
    
    return data_url

def test_doubao_api(image_path, prompt="请描述这张图片中的内容。"):
    """测试豆包API"""
    
    print("=" * 60)
    print("豆包API测试")
    print("=" * 60)
    
    # 检查API密钥
    if DOUBAO_API_KEY == "your-api-key-here":
        print("❌ 错误: 请先配置 DOUBAO_API_KEY 环境变量")
        return False
    
    # 检查图片文件
    if not os.path.exists(image_path):
        print(f"❌ 错误: 图片文件不存在: {image_path}")
        return False
    
    print(f"📷 图片路径: {image_path}")
    print(f"💬 提示词: {prompt}")
    print(f"🔑 API密钥: {DOUBAO_API_KEY[:10]}...")
    print(f"🌐 API地址: {DOUBAO_API_URL}")
    print(f"🤖 模型: {DOUBAO_MODEL}")
    print()
    
    try:
        # 转换图片为base64
        print("⏳ 正在转换图片为base64...")
        image_data_url = convert_image_to_base64(image_path)
        print(f"✓ 图片转换成功 (大小: {len(image_data_url)} 字符)")
        print()
        
        # 构建请求
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DOUBAO_API_KEY}"
        }
        
        payload = {
            "model": DOUBAO_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ]
        }
        
        # 发送请求
        print("⏳ 正在调用豆包API...")
        response = requests.post(DOUBAO_API_URL, headers=headers, json=payload, timeout=30)
        
        # 检查响应
        if response.status_code == 200:
            print("✓ API调用成功!")
            print()
            
            result = response.json()
            
            # 显示响应信息
            print("📊 响应信息:")
            print(f"  - 模型: {result.get('model', 'N/A')}")
            print(f"  - ID: {result.get('id', 'N/A')}")
            
            if 'usage' in result:
                usage = result['usage']
                print(f"  - Token使用:")
                print(f"    * 输入: {usage.get('prompt_tokens', 0)}")
                print(f"    * 输出: {usage.get('completion_tokens', 0)}")
                print(f"    * 总计: {usage.get('total_tokens', 0)}")
            
            print()
            
            # 显示分析结果
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                print("🎯 分析结果:")
                print("-" * 60)
                print(content)
                print("-" * 60)
                return True
            else:
                print("❌ 错误: 响应中没有分析结果")
                print(f"完整响应: {result}")
                return False
        else:
            print(f"❌ API调用失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print()
    print("豆包视觉API测试工具")
    print()
    
    # 查找测试图片
    test_images = []
    
    # 检查uploads目录
    if os.path.exists("uploads"):
        for file in os.listdir("uploads"):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join("uploads", file))
    
    if not test_images:
        print("❌ 未找到测试图片")
        print("请在 uploads 目录中放置一张图片,或指定图片路径")
        print()
        print("使用方法:")
        print("  python test_doubao_api.py [图片路径]")
        return
    
    # 使用第一张图片进行测试
    test_image = test_images[0]
    
    # 运行测试
    success = test_doubao_api(test_image)
    
    print()
    if success:
        print("✅ 测试通过! 豆包API集成正常工作")
    else:
        print("❌ 测试失败! 请检查配置和网络连接")
    print()

if __name__ == "__main__":
    import sys
    
    # 如果提供了命令行参数,使用指定的图片
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_doubao_api(image_path)
    else:
        main()
