import os
print("[INFO] 正在初始化 Python 环境并加载依赖...")
import json
import warnings

# Suppress TensorFlow logging (1: INFO, 2: WARNING, 3: ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from datetime import datetime
import asyncio
import cv2 as cv
import numpy as np
import mediapipe as mp
import time
from collections import deque, Counter
import threading
import queue
import requests
import dashscope
from pathlib import Path

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==================== 全局变量 ====================
# MediaPipe 模型
mp_hands = None
hands = None

# 线程间通信队列
frame_queue = queue.Queue(maxsize=2)  # 采集 → 识别
result_queue = queue.Queue(maxsize=1)  # 识别 → 编码

# 线程安全的共享变量
latest_frame = None  # 最新编码后的 JPEG 帧
latest_frame_lock = threading.Lock()

latest_raw_frame = None  # 最新原始帧 (用于截图)
latest_raw_frame_lock = threading.Lock()

# 截图功能配置
STILL_THRESHOLD = 10  # 静止检测阈值
STILL_DURATION = 2.0  # 静止持续时间(秒)
HISTORY_LENGTH = 30   # 位置历史记录长度

# 截图状态机状态
STATE_IDLE = "idle"
STATE_DETECTING = "detecting"
STATE_FRAMING = "framing"
STATE_WAITING_CAPTURE = "waiting_capture"
STATE_CAPTURED = "captured"

# 截图功能全局变量
screenshot_state = STATE_IDLE
screenshot_state_lock = threading.Lock()
index_finger_history = deque(maxlen=HISTORY_LENGTH)
still_start_time = None
frame_point1 = None
frame_point2 = None
screenshot_image = None
current_frame_for_screenshot = None
last_screenshot_path = None
gesture_history = deque(maxlen=8)  # 时序滤波缓冲区

# AI 分析相关
api_key = os.getenv("DASHSCOPE_API_KEY")
doubao_api_key = os.getenv("DOUBAO_API_KEY")  # 豆包API密钥
doubao_api_url = os.getenv("DOUBAO_API_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")  # 豆包API地址
doubao_model = os.getenv("DOUBAO_MODEL", "doubao-1.5-vision-pro-250328")  # 豆包模型名称
current_api_provider = "qwen"  # 当前使用的API提供商: "qwen" 或 "doubao"
current_api_provider_lock = threading.Lock()
latest_ai_response = ""
latest_ai_response_lock = threading.Lock()
selected_prompt_text = "请描述这张图片中的内容。"
PROMPTS_FILE = "prompts.json"

# 应用元数据配置
APP_NAME = "一框框系统 © 2026"
APP_VERSION = "1.2.0"
APP_COPYRIGHT = "作者:禾禾椰"
PROJECT_URL = "https://www.douyin.com/user/MS4wLjABAAAAOyqvejBV5f3GmXNbmeCmkCRQJ84Lcluy1uMeWwKa7o0"

# 摄像头运行状态
camera_running = True
capture_thread_obj = None
recognition_thread_obj = None
encoding_thread_obj = None

# 画面变换全局变量
rotation_angle = 180
is_mirror = True

# 性能监控
fps_counter = 0
fps_start_time = time.time()
current_fps = 0.0
fps_lock = threading.Lock()

# ==================== 辅助函数 ====================
def get_upload_policy(api_key, model_name):
    """获取文件上传凭证"""
    url = "https://dashscope.aliyuncs.com/api/v1/uploads"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    params = {
        "action": "getPolicy",
        "model": model_name
    }
    
    response = requests.get(url, headers=headers, params=params, timeout=5)
    if response.status_code != 200:
        raise Exception(f"Failed to get upload policy: {response.text}")
    
    return response.json()['data']

def upload_file_to_oss(policy_data, file_path):
    """将文件上传到临时存储OSS"""
    file_name = Path(file_path).name
    key = f"{policy_data['upload_dir']}/{file_name}"
    
    with open(file_path, 'rb') as file:
        files = {
            'OSSAccessKeyId': (None, policy_data['oss_access_key_id']),
            'Signature': (None, policy_data['signature']),
            'policy': (None, policy_data['policy']),
            'x-oss-object-acl': (None, policy_data['x_oss_object_acl']),
            'x-oss-forbid-overwrite': (None, policy_data['x_oss_forbid_overwrite']),
            'key': (None, key),
            'success_action_status': (None, '200'),
            'file': (file_name, file)
        }
        
        response = requests.post(policy_data['upload_host'], files=files, timeout=15)
        if response.status_code != 200:
            raise Exception(f"Failed to upload file: {response.text}")
    
    return f"oss://{key}"

def upload_file_and_get_url(api_key, model_name, file_path):
    """上传文件并获取URL"""
    policy_data = get_upload_policy(api_key, model_name)
    oss_url = upload_file_to_oss(policy_data, file_path)
    return oss_url

def call_qwen_vl(api_key, oss_url):
    """调用 Qwen-VL 模型进行图像理解"""
    global selected_prompt_text
    messages = [
        {
            "role": "system",
            "content": [{"text": "你是一个专业的图像分析助手。"}]
        },
        {
            "role": "user",
            "content": [
                {"image": oss_url},
                {"text": selected_prompt_text}]
        }]

    response = dashscope.MultiModalConversation.call(
        api_key=api_key,
        model='qwen-vl-plus',
        messages=messages
    )
    return response

def call_doubao_vl(api_key, image_url):
    """调用豆包视觉模型进行图像理解"""
    global selected_prompt_text, doubao_api_url, doubao_model
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 构建请求体
    payload = {
        "model": doubao_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": selected_prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ]
    }
    
    response = requests.post(doubao_api_url, headers=headers, json=payload, timeout=30)
    
    if response.status_code != 200:
        raise Exception(f"豆包API调用失败: {response.status_code} - {response.text}")
    
    return response.json()

def convert_local_image_to_base64(file_path):
    """将本地图片转换为base64编码的data URL"""
    import base64
    
    with open(file_path, 'rb') as f:
        image_data = f.read()
    
    # 获取文件扩展名
    ext = Path(file_path).suffix.lower()
    mime_type = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
    
    # 转换为base64
    base64_data = base64.b64encode(image_data).decode('utf-8')
    data_url = f"data:{mime_type};base64,{base64_data}"
    
    return data_url

def auto_upload_task(file_path):
    """后台上传任务,避免阻塞识别主循环"""
    global api_key, doubao_api_key, current_api_provider, latest_ai_response
    
    # 获取当前API提供商
    with current_api_provider_lock:
        provider = current_api_provider
    
    # 检查API密钥
    if provider == "qwen" and not api_key:
        print("自动上传取消: 未配置通义千问 API Key")
        with latest_ai_response_lock:
            latest_ai_response = "错误: 未配置通义千问 API Key"
        return
    
    if provider == "doubao" and not doubao_api_key:
        print("自动上传取消: 未配置豆包 API Key")
        with latest_ai_response_lock:
            latest_ai_response = "错误: 未配置豆包 API Key"
        return
    
    try:
        with latest_ai_response_lock:
            latest_ai_response = "正在分析中..."
        
        if provider == "qwen":
            # 使用通义千问API
            print(f"[通义千问] 开始上传云端: {file_path}")
            model_name = "qwen-vl-plus"
            oss_url = upload_file_and_get_url(api_key, model_name, file_path)
            print(f"[通义千问] 云端上传成功: {oss_url}")
            
            # 调用通义千问模型进行分析
            ai_res = call_qwen_vl(api_key, oss_url)
            if ai_res.status_code == 200:
                result = ai_res.output.choices[0].message.content[0]['text']
                with latest_ai_response_lock:
                    latest_ai_response = result
                print(f"[通义千问] AI 分析结果: {result}")
            else:
                error_msg = f"通义千问分析失败: {ai_res.message}"
                with latest_ai_response_lock:
                    latest_ai_response = error_msg
                print(f"[通义千问] AI 分析失败: {ai_res.code} - {ai_res.message}")
        
        elif provider == "doubao":
            # 使用豆包API
            print(f"[豆包] 开始分析图片: {file_path}")
            
            # 将本地图片转换为base64 data URL
            image_data_url = convert_local_image_to_base64(file_path)
            print(f"[豆包] 图片已转换为base64格式")
            
            # 调用豆包模型进行分析
            ai_res = call_doubao_vl(doubao_api_key, image_data_url)
            
            # 解析豆包响应
            if 'choices' in ai_res and len(ai_res['choices']) > 0:
                result = ai_res['choices'][0]['message']['content']
                with latest_ai_response_lock:
                    latest_ai_response = result
                print(f"[豆包] AI 分析结果: {result}")
            else:
                error_msg = f"豆包分析失败: 响应格式异常"
                with latest_ai_response_lock:
                    latest_ai_response = error_msg
                print(f"[豆包] AI 分析失败: {ai_res}")
        
    except Exception as e:
        error_msg = f"[{provider}] 分析出错: {str(e)}"
        with latest_ai_response_lock:
            latest_ai_response = error_msg
        print(f"[{provider}] 分析失败: {e}")

def apply_transforms(frame):
    """应用旋转和镜像变换"""
    if rotation_angle == 90:
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        frame = cv.rotate(frame, cv.ROTATE_180)
    elif rotation_angle == 270:
        frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    
    if is_mirror:
        frame = cv.flip(frame, 1)
    
    return frame

def calculate_angle(a, b, c):
    """计算三个点形成的夹角 (0-180度)"""
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

# ==================== 截图相关函数 ====================
def is_finger_still(current_pos):
    """检测食指是否静止"""
    if len(index_finger_history) < 5:
        return False
    
    recent_positions = list(index_finger_history)[-5:]
    avg_x = sum(p[0] for p in recent_positions) / len(recent_positions)
    avg_y = sum(p[1] for p in recent_positions) / len(recent_positions)
    
    distance = np.sqrt((current_pos[0] - avg_x)**2 + (current_pos[1] - avg_y)**2)
    return distance < STILL_THRESHOLD

def update_screenshot_state(image, hand_sign_id, index_finger_pos):
    """更新截图状态机"""
    global screenshot_state, still_start_time, frame_point1, frame_point2
    global screenshot_image, current_frame_for_screenshot
    
    current_time = time.time()
    
    with screenshot_state_lock:
        # 在 FRAMING 状态时持续保存最新帧
        if screenshot_state in [STATE_FRAMING, STATE_WAITING_CAPTURE]:
            current_frame_for_screenshot = image.copy()
        
        if screenshot_state == STATE_IDLE:
            if hand_sign_id == 2:  # Point gesture
                screenshot_state = STATE_DETECTING
                still_start_time = None
                index_finger_history.clear()
        
        elif screenshot_state == STATE_DETECTING:
            if hand_sign_id != 2:
                screenshot_state = STATE_IDLE
                still_start_time = None
                index_finger_history.clear()
            else:
                index_finger_history.append(index_finger_pos)
                
                if is_finger_still(index_finger_pos):
                    if still_start_time is None:
                        still_start_time = current_time
                    else:
                        elapsed = current_time - still_start_time
                        if elapsed >= STILL_DURATION:
                            frame_point1 = index_finger_pos
                            frame_point2 = index_finger_pos
                            screenshot_state = STATE_FRAMING
                            still_start_time = None
                            # 保存第一帧
                            current_frame_for_screenshot = image.copy()
                else:
                    still_start_time = None
        
        elif screenshot_state == STATE_FRAMING:
            if hand_sign_id != 2:
                screenshot_state = STATE_IDLE
                still_start_time = None
                frame_point1 = None
                frame_point2 = None
                index_finger_history.clear()
                current_frame_for_screenshot = None
            else:
                frame_point2 = index_finger_pos
                index_finger_history.append(index_finger_pos)
                
                if is_finger_still(index_finger_pos):
                    if still_start_time is None:
                        still_start_time = current_time
                    else:
                        elapsed = current_time - still_start_time
                        if elapsed >= STILL_DURATION:
                            screenshot_state = STATE_WAITING_CAPTURE
                else:
                    still_start_time = None
        
        elif screenshot_state == STATE_WAITING_CAPTURE:
            if frame_point1 and frame_point2 and current_frame_for_screenshot is not None:
                screenshot_image = capture_screenshot(current_frame_for_screenshot, frame_point1, frame_point2)
                if screenshot_image is not None:
                    screenshot_state = STATE_CAPTURED
                    # 触发后台上传
                    if last_screenshot_path:
                        threading.Thread(target=auto_upload_task, args=(last_screenshot_path,), daemon=True).start()
                else:
                    print("截图失败,重置状态")
                    screenshot_state = STATE_IDLE
                    current_frame_for_screenshot = None
            else:
                print(f"截图条件不满足: pt1={frame_point1}, pt2={frame_point2}, frame={current_frame_for_screenshot is not None}")
                screenshot_state = STATE_IDLE
                current_frame_for_screenshot = None

def capture_screenshot(image, pt1, pt2):
    """执行截图,裁剪指定区域"""
    global last_screenshot_path
    
    # 检查图像是否为空
    if image is None or image.size == 0:
        print("警告: 截图失败,图像为空")
        return None
    
    x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
    x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
    
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # 确保裁剪区域有效
    if x2 <= x1 or y2 <= y1:
        print(f"警告: 截图区域无效 ({x1},{y1}) -> ({x2},{y2})")
        return None
    
    cropped = image[y1:y2, x1:x2]
    
    # 再次检查裁剪后的图像
    if cropped is None or cropped.size == 0:
        print("警告: 裁剪后的图像为空")
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"screenshot_{timestamp}.jpg"
    filepath = os.path.join(UPLOAD_DIR, filename)
    cv.imwrite(filepath, cropped)
    last_screenshot_path = filepath
    
    print(f"截图已保存: {filename}")
    return cropped

def draw_landmarks(image, landmark_point):
    """在图像上绘制手部关键点和骨架"""
    if len(landmark_point) > 0:
        connections = [
            (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12), (13, 14), (14, 15),
            (15, 16), (17, 18), (18, 19), (19, 20),
            (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0)
        ]
        
        img_width, img_height = image.shape[1], image.shape[0]
        points = []
        for lp in landmark_point:
            points.append((int(lp[0] * img_width), int(lp[1] * img_height)))

        # 绘制骨架连线
        for start_idx, end_idx in connections:
            cv.line(image, points[start_idx], points[end_idx], (0, 0, 0), 6)
            cv.line(image, points[start_idx], points[end_idx], (255, 255, 255), 2)

        # 绘制关键点
        for i, pt in enumerate(points):
            size = 8 if i in [4, 8, 12, 16, 20] else 5
            cv.circle(image, pt, size, (255, 255, 255), -1)
            cv.circle(image, pt, size, (0, 0, 0), 1)

    return image

def classify_gesture(image: np.ndarray):
    """检测手部并提取食指位置 (增强版: 角度分析 + 时序滤波)"""
    global frame_point1, frame_point2, screenshot_state
    
    # 转换为RGB用于MediaPipe处理
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # 性能优化:缩放图像进行 AI 推理 (320x180 更快)
    h, w = image_rgb.shape[:2]
    scale_width = 320  # 降低到 320 (原来是 480)
    image_small = cv.resize(image_rgb, (scale_width, int(h * (scale_width / w))), interpolation=cv.INTER_LINEAR)
    image_small.flags.writeable = False
    results = hands.process(image_small)
    
    raw_gesture_id = -1  # 本帧的原始识别结果
    index_finger_pos = None
    landmarks_list = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y])
            
            # 将归一化坐标转换为像素坐标(用于角度计算)
            # 注意: 计算角度最好用像素坐标以保持比例,或者统一纵横比
            # 这里简单处理,直接用归一化坐标计算角度也行,但会有透视误差. 
            # 既然是简单判断,先用归一化坐标.
            
            # 定义关键点索引
            # 0: wrist, 1-4: thumb, 5-8: index, 9-12: middle, 13-16: ring, 17-20: pinky
            
            # 计算各手指弯曲角度 (关节 0, 1, 2) -> (base, mid, tip) is wrong.
            # Correct joints for flexion:
            # Thumb: 2-3-4 (IP joint) seems useful, but usually verify 1-2-4 or angle between 2-3-4.
            # Others: 5-6-7 (MCP-PIP-DIP) is not straight line. 
            # Usually check angle at PIP (e.g. 5-6-7) or simply dist tip to palm.
            
            # 让我们使用更鲁棒的角度判断:
            # MCP(root) -> PIP(mid) -> DIP(distal) is usually straight when open.
            # We measure angle at PIP (e.g. indices 5, 6, 7)
            
            # Thumb angle: 2, 3, 4
            angle_thumb = calculate_angle(landmarks[2], landmarks[3], landmarks[4])
            # Index angle: 5, 6, 7
            angle_index = calculate_angle(landmarks[5], landmarks[6], landmarks[7])
            # Middle angle: 9, 10, 11
            angle_middle = calculate_angle(landmarks[9], landmarks[10], landmarks[11])
            # Ring angle: 13, 14, 15
            angle_ring = calculate_angle(landmarks[13], landmarks[14], landmarks[15])
            # Pinky angle: 17, 18, 19
            angle_pinky = calculate_angle(landmarks[17], landmarks[18], landmarks[19])
            
            # 判断手指伸直/弯曲 (阈值宽松一点, >160 直, <100 弯)
            is_thumb_open = angle_thumb > 150 # 拇指比较特殊
            is_index_open = angle_index > 160
            is_middle_open = angle_middle > 160
            is_ring_open = angle_ring > 160
            is_pinky_open = angle_pinky > 160
            
            # 获取食指尖端位置
            img_width, img_height = image.shape[1], image.shape[0]
            index_finger_pos = [
                int(landmarks[8][0] * img_width),
                int(landmarks[8][1] * img_height)
            ]
            landmarks_list = landmarks

            # === 核心手势逻辑 ===
            
            # 1. Pointer (指指点点)
            # 要求: 食指伸直, 其他三指(中无小)弯曲. 拇指随意(通常弯曲或自然放置)
            if is_index_open and (not is_middle_open) and (not is_ring_open) and (not is_pinky_open):
                # 额外校验: 指尖方向向上 (y坐标 8 < 6)
                if landmarks[8][1] < landmarks[6][1]: 
                     raw_gesture_id = 2
            break
    
    # === 时序滤波 (Debouncing) ===
    gesture_history.append(raw_gesture_id)
    
    # 统计缓冲区中最多的状态
    most_common = Counter(gesture_history).most_common(1)
    if not most_common:
        stable_gesture_id = -1
    else:
        # 只有当缓冲区中 75% 以上一致时才切换状态 (8个里至少6个)
        mode_id, count = most_common[0]
        if count >= 6:
            stable_gesture_id = mode_id
        else:
            # 不稳定时保持上一帧状态(或者保持 -1), 这里选择保持 -1 更安全(宁缺毋滥)
            stable_gesture_id = -1

            # 另一种策略: 如果不稳定, 沿用上一次的 stable 状态? 
            # 简单起见, 不稳定就视为 no hand, 避免乱跳
    
    # 更新截图状态机 (使用滤波后的结果)
    update_screenshot_state(image, stable_gesture_id, index_finger_pos)
    
    # 绘制
    if landmarks_list is not None:
        img_width, img_height = image.shape[1], image.shape[0]
        key_points = [4, 8, 12, 16, 20]
        # 根据识别结果改变绘制颜色
        color = (0, 255, 0) # 默认绿
        if stable_gesture_id == 2: color = (0, 255, 255) # 黄 Pointer
        
        for i in key_points:
            pt = (int(landmarks_list[i][0] * img_width), int(landmarks_list[i][1] * img_height))
            cv.circle(image, pt, 6, color, -1)
    
    # 绘制截图框
    with screenshot_state_lock:
        if frame_point1 and frame_point2 and screenshot_state in [STATE_DETECTING, STATE_FRAMING]:
            cv.rectangle(image, tuple(frame_point1), tuple(frame_point2), (0, 255, 0), 2)
            cv.circle(image, tuple(frame_point1), 6, (255, 0, 0), -1)
    
    # 更新全局
    gesture_text = "no hand"
    if stable_gesture_id == 2:
        gesture_text = "pointer"
    
    return gesture_text, image

# ==================== 三线程架构 ====================
def capture_thread():
    """采集线程:高速采集原始帧 (目标 30 FPS)"""
    global latest_raw_frame, current_fps, fps_counter, fps_start_time
    
    print("尝试打开USB摄像头...")
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    
    if not cap.isOpened():
        cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        cap = cv.VideoCapture(1)

    if not cap.isOpened():
        print("错误: 无法打开任何USB摄像头")
        return

    # 设置分辨率和帧率 (降低分辨率以提升性能)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)   # 从 1280 降低到 640
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)  # 从 720 降低到 480
    cap.set(cv.CAP_PROP_FPS, 30)
    print("USB摄像头已打开 (采集线程)")

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        
        # 应用变换
        frame = apply_transforms(frame)
        
        # 更新最新原始帧 (用于截图)
        with latest_raw_frame_lock:
            latest_raw_frame = frame.copy()
        
        # 非阻塞放入队列
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            # 清空队列,只保留最新帧
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            frame_queue.put_nowait(frame)
        
        # FPS 统计
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            with fps_lock:
                current_fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        time.sleep(0.001)  # 极小延迟
        
    cap.release()
    print("USB摄像头已释放 (采集线程)")

def recognition_thread():
    """识别线程:手势识别 (处理所有帧)"""
    
    while camera_running:
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        try:
            # 手势识别
            _, debug_image = classify_gesture(frame)
            
            # 更新结果队列
            try:
                result_queue.put_nowait(debug_image)
            except queue.Full:
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
                result_queue.put_nowait(debug_image)
        except Exception as e:
            print(f"识别线程出错: {e}")

def encoding_thread():
    """编码线程:JPEG 编码 (按需触发)"""
    global latest_frame
    last_encoded_id = None
    
    while camera_running:
        try:
            debug_image = result_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        # 缓存优化:避免重复编码
        frame_id = id(debug_image)
        if frame_id == last_encoded_id:
            continue
        
        try:
            # JPEG 编码 (降低质量以提升速度)
            encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 60]  # 从 80 降低到 60
            _, encoded = cv.imencode('.jpg', debug_image, encode_param)
            
            # 更新全局变量
            with latest_frame_lock:
                latest_frame = encoded.tobytes()
            
            last_encoded_id = frame_id
        except Exception as e:
            print(f"编码线程出错: {e}")

# ==================== FastAPI 应用 ====================
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    global mp_hands, hands, camera_running
    global capture_thread_obj, recognition_thread_obj, encoding_thread_obj
    
    # 检查API密钥配置
    if not api_key:
        print("警告: 环境变量 DASHSCOPE_API_KEY 未配置,通义千问 AI 分析功能将不可用")
    if not doubao_api_key:
        print("警告: 环境变量 DOUBAO_API_KEY 未配置,豆包 AI 分析功能将不可用")
    if not api_key and not doubao_api_key:
        print("警告: 未配置任何 AI API 密钥,AI 分析功能将完全不可用")

    print("[INFO] 正在加载 MediaPipe 手势识别模型...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,   # 提高检测阈值
        min_tracking_confidence=0.6,    # 提高跟踪阈值
        model_complexity=1              # 提高模型复杂度 (0->1)
    )
    print("MediaPipe模型加载成功")

    # 预热模型
    print("[INFO] 正在预热 AI 模型...")
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    hands.process(dummy_frame)
    
    print("[INFO] 正在启动三线程架构...")
    camera_running = True
    
    # 启动采集线程
    capture_thread_obj = threading.Thread(target=capture_thread, daemon=True, name="CaptureThread")
    capture_thread_obj.start()
    
    # 启动识别线程
    recognition_thread_obj = threading.Thread(target=recognition_thread, daemon=True, name="RecognitionThread")
    recognition_thread_obj.start()
    
    # 启动编码线程
    encoding_thread_obj = threading.Thread(target=encoding_thread, daemon=True, name="EncodingThread")
    encoding_thread_obj.start()
    
    print("[INFO] 所有线程已启动,系统就绪!")

    try:
        yield
    finally:
        print("\n正在关闭服务...")
        camera_running = False
        
        # 等待线程退出
        for thread_obj in [capture_thread_obj, recognition_thread_obj, encoding_thread_obj]:
            if thread_obj and thread_obj.is_alive():
                thread_obj.join(timeout=2.0)
        
        if hands:
            hands.close()
            print("MediaPipe Hands模型已释放")
        print("服务已完全停止")

app = FastAPI(lifespan=lifespan)

# ==================== API 路由 ====================
@app.get("/")
async def root():
    """根路由,重定向到截图页面"""
    return FileResponse("screenshot.html")

@app.get("/screenshot.html")
async def screenshot_page():
    """单页应用 - 截图和AI分析"""
    return FileResponse("screenshot.html")

@app.websocket("/ws/video")
async def websocket_video_stream(websocket: WebSocket):
    """WebSocket 视频流推送接口"""
    await websocket.accept()
    last_frame_id = None
    last_sent_state = None
    
    try:
        while True:
            # 1. 检查截图状态，如果完成则发送通知
            current_state = None
            with screenshot_state_lock:
                current_state = screenshot_state
            
            if current_state == STATE_CAPTURED and last_sent_state != STATE_CAPTURED:
                await websocket.send_text("CAPTURED")
                last_sent_state = STATE_CAPTURED
            elif current_state == STATE_IDLE:
                last_sent_state = STATE_IDLE

            # 2. 发送视频帧
            current_frame = None
            with latest_frame_lock:
                if latest_frame is not None:
                    # 使用帧 ID 避免重复发送相同帧
                    frame_id = id(latest_frame)
                    if frame_id != last_frame_id:
                        current_frame = latest_frame
                        last_frame_id = frame_id
            
            if current_frame:
                await websocket.send_bytes(current_frame)
            
            # 等待约 33ms (30 FPS)
            await asyncio.sleep(0.033)
            
    except WebSocketDisconnect:
        print("[WebSocket] 客户端断开连接")
    except Exception as e:
        print(f"[WebSocket] 错误: {e}")

@app.get("/get_screenshot")
async def get_screenshot():
    """获取截图结果"""
    if screenshot_image is None:
        return Response(content=b"", status_code=404)
    
    _, buffer = cv.imencode('.jpg', screenshot_image)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

def reset_screenshot_logic():
    """内部重置逻辑"""
    global screenshot_state, still_start_time, frame_point1, frame_point2
    global screenshot_image, latest_ai_response, last_screenshot_path
    global current_frame_for_screenshot
    
    with screenshot_state_lock:
        screenshot_state = STATE_IDLE
        still_start_time = None
        frame_point1 = None
        frame_point2 = None
        screenshot_image = None
        current_frame_for_screenshot = None
        last_screenshot_path = None
        index_finger_history.clear()
    
    with latest_ai_response_lock:
        latest_ai_response = ""

@app.post("/reset_screenshot")
async def reset_screenshot():
    """重置截图状态接口"""
    reset_screenshot_logic()
    return JSONResponse(content={"message": "Screenshot state reset"})

@app.get("/get_ai_response")
async def get_ai_response():
    """获取最新的 AI 分析结果"""
    with latest_ai_response_lock:
        response = latest_ai_response or ""
    return JSONResponse(content={"analysis": response})

@app.post("/toggle_rotate")
async def toggle_rotate():
    """切换旋转角度"""
    global rotation_angle
    rotation_angle = (rotation_angle + 90) % 360
    reset_screenshot_logic()
    return JSONResponse(content={"angle": rotation_angle})

@app.post("/toggle_mirror")
async def toggle_mirror():
    """切换镜像状态"""
    global is_mirror
    is_mirror = not is_mirror
    reset_screenshot_logic()
    return JSONResponse(content={"mirror": is_mirror})

@app.get("/get_prompts")
async def get_prompts():
    """获取提示词列表"""
    if os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
            return JSONResponse(content=json.load(f))
    return JSONResponse(content=[])

@app.post("/select_prompt")
async def select_prompt(request: Request):
    """设置当前使用的提示词"""
    global selected_prompt_text
    data = await request.json()
    prompt_content = data.get("content")
    if prompt_content:
        selected_prompt_text = prompt_content
        print(f"提示词已切换为: {selected_prompt_text}")
        return JSONResponse(content={"message": "Prompt updated"})
    return JSONResponse(content={"error": "Invalid prompt"}, status_code=400)

@app.get("/get_api_provider")
async def get_api_provider():
    """获取当前使用的API提供商"""
    with current_api_provider_lock:
        provider = current_api_provider
    
    return JSONResponse(content={
        "provider": provider,
        "qwen_configured": bool(api_key),
        "doubao_configured": bool(doubao_api_key),
        "doubao_model": doubao_model
    })

@app.post("/set_api_provider")
async def set_api_provider(request: Request):
    """设置当前使用的API提供商"""
    global current_api_provider
    
    data = await request.json()
    provider = data.get("provider")
    
    if provider not in ["qwen", "doubao"]:
        return JSONResponse(
            content={"error": "Invalid provider. Must be 'qwen' or 'doubao'"},
            status_code=400
        )
    
    # 检查对应的API密钥是否配置
    if provider == "qwen" and not api_key:
        return JSONResponse(
            content={"error": "通义千问 API Key 未配置"},
            status_code=400
        )
    
    if provider == "doubao" and not doubao_api_key:
        return JSONResponse(
            content={"error": "豆包 API Key 未配置"},
            status_code=400
        )
    
    with current_api_provider_lock:
        current_api_provider = provider
    
    provider_name = "通义千问" if provider == "qwen" else "豆包"
    print(f"API提供商已切换为: {provider_name}")
    
    return JSONResponse(content={
        "message": f"API提供商已切换为 {provider_name}",
        "provider": provider
    })

@app.get("/app_info")
async def get_app_info():
    """获取应用版本和版权信息"""
    return JSONResponse(content={
        "name": APP_NAME,
        "version": APP_VERSION,
        "copyright": APP_COPYRIGHT,
        "url": PROJECT_URL
    })

@app.get("/performance_stats")
async def get_performance_stats():
    """获取性能统计信息 (新增接口)"""
    with fps_lock:
        fps = current_fps
    
    return JSONResponse(content={
        "fps": fps,
        "frame_queue_size": frame_queue.qsize(),
        "result_queue_size": result_queue.qsize(),
        "threads_alive": {
            "capture": capture_thread_obj.is_alive() if capture_thread_obj else False,
            "recognition": recognition_thread_obj.is_alive() if recognition_thread_obj else False,
            "encoding": encoding_thread_obj.is_alive() if encoding_thread_obj else False
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
