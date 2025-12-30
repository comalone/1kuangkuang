import os
print("[INFO] 正在初始化 Python 环境并加载依赖...")
import json
import warnings

# Suppress TensorFlow logging (1: INFO, 2: WARNING, 3: ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, FileResponse
from datetime import datetime
import cv2 as cv
import numpy as np
import mediapipe as mp
import time
from collections import deque
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
current_gesture = "no hand"
current_gesture_lock = threading.Lock()

# AI 分析相关
api_key = os.getenv("DASHSCOPE_API_KEY")
latest_ai_response = ""
latest_ai_response_lock = threading.Lock()
selected_prompt_text = "请描述这张图片中的内容。"
PROMPTS_FILE = "prompts.json"

# 应用元数据配置
APP_NAME = "一框框系统 © 2026"
APP_VERSION = "1.1.3"
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

def auto_upload_task(file_path):
    """后台上传任务,避免阻塞识别主循环"""
    global api_key, latest_ai_response
    if not api_key:
        print("自动上传取消: 未配置 api_key")
        return
    try:
        with latest_ai_response_lock:
            latest_ai_response = "正在分析中..."
        print(f"开始自动上传云端: {file_path}")
        model_name = "qwen-vl-plus"
        oss_url = upload_file_and_get_url(api_key, model_name, file_path)
        print(f"云端上传成功: {oss_url}")
        
        # 调用 AI 模型进行分析
        ai_res = call_qwen_vl(api_key, oss_url)
        if ai_res.status_code == 200:
            result = ai_res.output.choices[0].message.content[0]['text']
            with latest_ai_response_lock:
                latest_ai_response = result
            print(f"AI 分析结果: {result}")
        else:
            error_msg = f"AI 分析失败: {ai_res.message}"
            with latest_ai_response_lock:
                latest_ai_response = error_msg
            print(f"AI 分析失败: {ai_res.code} - {ai_res.message}")
    except Exception as e:
        error_msg = f"上传或分析出错: {str(e)}"
        with latest_ai_response_lock:
            latest_ai_response = error_msg
        print(f"云端上传失败: {e}")

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
    """检测手部并提取食指位置"""
    global frame_point1, frame_point2, screenshot_state, current_gesture
    
    # 转换为RGB用于MediaPipe处理
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # 性能优化:缩放图像进行 AI 推理 (320x180 更快)
    h, w = image_rgb.shape[:2]
    scale_width = 320  # 降低到 320 (原来是 480)
    image_small = cv.resize(image_rgb, (scale_width, int(h * (scale_width / w))), interpolation=cv.INTER_LINEAR)
    image_small.flags.writeable = False
    results = hands.process(image_small)
    
    hand_sign_id = -1
    index_finger_pos = None
    landmarks_list = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y])

            # 手势识别逻辑
            dist_4_8 = np.sqrt((landmarks[4][0]-landmarks[8][0])**2 + (landmarks[4][1]-landmarks[8][1])**2)
            
            if dist_4_8 < 0.05:
                hand_sign_id = 3  # OK手势
            elif landmarks[8][1] < landmarks[5][1]:  # 食指伸出
                hand_sign_id = 2  # Pointer手势
            else:
                hand_sign_id = -1
            
            # 获取食指尖端位置
            img_width, img_height = image.shape[1], image.shape[0]
            index_finger_pos = [
                int(landmarks[8][0] * img_width),
                int(landmarks[8][1] * img_height)
            ]
            
            landmarks_list = landmarks
            break
    
    # 更新截图状态机
    update_screenshot_state(image, hand_sign_id, index_finger_pos)
    
    # 简化绘制:只绘制指尖关键点 (不绘制骨架,大幅提升性能)
    if landmarks_list is not None:
        img_width, img_height = image.shape[1], image.shape[0]
        # 只绘制5个指尖
        key_points = [4, 8, 12, 16, 20]
        for i in key_points:
            pt = (int(landmarks_list[i][0] * img_width), int(landmarks_list[i][1] * img_height))
            cv.circle(image, pt, 6, (0, 255, 0), -1)  # 绿色实心圆
    
    # 绘制截图框定区域
    with screenshot_state_lock:
        if frame_point1 and frame_point2 and screenshot_state in [STATE_DETECTING, STATE_FRAMING]:
            cv.rectangle(image, tuple(frame_point1), tuple(frame_point2), (0, 255, 0), 2)
            cv.circle(image, tuple(frame_point1), 6, (255, 0, 0), -1)
    
    # 更新全局手势变量
    gesture_text = "no hand"
    if hand_sign_id == 3:
        gesture_text = "ok"
    elif hand_sign_id == 2:
        gesture_text = "pointer"
    
    with current_gesture_lock:
        current_gesture = gesture_text
        
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
    
    if not api_key:
        print("警告: 环境变量 DASHSCOPE_API_KEY 未配置,AI 分析功能将不可用")

    print("[INFO] 正在加载 MediaPipe 手势识别模型...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0  # 轻量级模型
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
    """截图页面"""
    return FileResponse("screenshot.html")

@app.get("/screenshot_result.html")
async def screenshot_result_page():
    """截图结果页面"""
    return FileResponse("screenshot_result.html")

@app.get("/video_feed")
async def video_feed():
    """提供给前端显示的视频流接口"""
    with latest_frame_lock:
        if latest_frame is None:
            return Response(content=b"", status_code=200, media_type="image/jpeg")
        return Response(content=latest_frame, media_type="image/jpeg")

@app.get("/screenshot_status")
async def screenshot_status():
    """获取当前截图状态"""
    with screenshot_state_lock:
        state = screenshot_state
        pt1 = frame_point1
        pt2 = frame_point2
    
    with current_gesture_lock:
        gesture = current_gesture
    
    return JSONResponse(content={
        "state": state,
        "point1": pt1,
        "point2": pt2,
        "gesture": gesture,
        "has_screenshot": screenshot_image is not None
    })

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
    global current_gesture, current_frame_for_screenshot
    
    with screenshot_state_lock:
        screenshot_state = STATE_IDLE
        still_start_time = None
        frame_point1 = None
        frame_point2 = None
        screenshot_image = None
        current_frame_for_screenshot = None
        last_screenshot_path = None
        index_finger_history.clear()
    
    with current_gesture_lock:
        current_gesture = "no hand"
    
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
