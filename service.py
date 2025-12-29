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
import requests
import dashscope
from pathlib import Path

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 初始化变量
mp_hands = None
hands = None
latest_frame = None  # 用于缓存最新的处理结果

# 截图功能配置
STILL_THRESHOLD = 20  # 提升阈值以适配 1280x720 高分辨率下的像素密度
STILL_DURATION = 1.0  # 静止持续时间(秒)
HISTORY_LENGTH = 30   # 位置历史记录长度

# 截图状态机状态
STATE_IDLE = "idle"  # 空闲状态
STATE_DETECTING = "detecting"  # 检测中
STATE_FRAMING = "framing"  # 框定中
STATE_WAITING_CAPTURE = "waiting_capture"  # 等待截图
STATE_CAPTURED = "captured"  # 已截图

# 截图功能全局变量
screenshot_state = STATE_IDLE
index_finger_history = deque(maxlen=HISTORY_LENGTH)
still_start_time = None
frame_point1 = None  # 框定起始点
frame_point2 = None  # 框定结束点
screenshot_image = None  # 截图结果
current_frame_for_screenshot = None  # 当前帧(用于截图)
last_screenshot_path = None  # 最近一次截图的路径
current_gesture = "no hand"  # 当前识别到的手势
api_key = os.getenv("DASHSCOPE_API_KEY")
latest_ai_response = "" # 最新 AI 分析结果
selected_prompt_text = "请描述这张图片中的内容。" # 默认提示词
PROMPTS_FILE = "prompts.json"

# 应用元数据配置
APP_NAME = "一顿框系统 © 2026"
APP_VERSION = "1.0.2"
APP_COPYRIGHT = "作者：禾禾椰"
PROJECT_URL = "https://www.douyin.com/user/MS4wLjABAAAAOyqvejBV5f3GmXNbmeCmkCRQJ84Lcluy1uMeWwKa7o0"

# 摄像头运行状态全局变量
camera_running = True
usb_thread = None

# 画面变换全局变量
rotation_angle = 180
is_mirror = True

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
    # 1. 获取上传凭证
    policy_data = get_upload_policy(api_key, model_name) 
    # 2. 上传文件到OSS
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
    """后台上传任务，避免阻塞识别主循环"""
    global api_key, latest_ai_response
    if not api_key:
        print("自动上传取消: 未配置 api_key")
        return
    try:
        latest_ai_response = "正在分析中..."
        print(f"开始自动上传云端: {file_path}")
        model_name = "qwen-vl-plus"
        oss_url = upload_file_and_get_url(api_key, model_name, file_path)
        print(f"云端上传成功: {oss_url}")
        
        # 调用 AI 模型进行分析
        ai_res = call_qwen_vl(api_key, oss_url)
        if ai_res.status_code == 200:
            latest_ai_response = ai_res.output.choices[0].message.content[0]['text']
            print(f"AI 分析结果: {latest_ai_response}")
        else:
            latest_ai_response = f"AI 分析失败: {ai_res.message}"
            print(f"AI 分析失败: {ai_res.code} - {ai_res.message}")
    except Exception as e:
        latest_ai_response = f"上传或分析出错: {str(e)}"
        print(f"云端上传失败: {e}")

async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    global mp_hands, hands, camera_running, usb_thread
    
    if not api_key:
        print("警告: 环境变量 DASHSCOPE_API_KEY 未配置，AI 分析功能将不可用")

    print("[INFO] 正在加载 MediaPipe 手势识别模型...")
    mp_hands = mp.solutions.hands
    # 优化性能配置:使用最快的模型
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # 只检测一只手
        min_detection_confidence=0.5,  # 降低检测阈值,提升速度
        min_tracking_confidence=0.5,  # 降低跟踪阈值
        model_complexity=0  # 使用轻量级模型，显著提升处理速度
    )
    print("MediaPipe模型加载成功")

    # 预热模型：避免第一次识别时因初始化 TFLite 导致画面卡顿
    print("[INFO] 正在预热 AI 模型...")
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    hands.process(dummy_frame)
    
    print("[INFO] 正在启动摄像头采集线程...")
    usb_thread = threading.Thread(target=usb_camera_loop, daemon=True)
    usb_thread.start()

    # 在应用关闭时释放模型资源
    try:
        yield
    finally:
        print("\n正在关闭服务...")
        camera_running = False  # 停止USB摄像头线程
        if usb_thread and usb_thread.is_alive():
            print("正在等待USB摄像头线程释放资源...")
            usb_thread.join(timeout=2.0)
            
        if hands:
            hands.close()
            print("MediaPipe Hands模型已释放")
        print("服务已完全停止")

app = FastAPI(lifespan=lifespan)

# 添加根路由,重定向到截图页面
@app.get("/")
async def root():
    """根路由,重定向到截图页面"""
    return FileResponse("screenshot.html")

# 提供HTML页面
@app.get("/screenshot.html")
async def screenshot_page():
    """截图页面"""
    return FileResponse("screenshot.html")

@app.get("/screenshot_result.html")
async def screenshot_result_page():
    """截图结果页面"""
    return FileResponse("screenshot_result.html")

def is_finger_still(current_pos):
    """检测食指是否静止"""
    if len(index_finger_history) < 5:  # 需要至少5帧数据
        return False
    
    # 计算最近几帧的平均位置
    recent_positions = list(index_finger_history)[-5:]
    avg_x = sum(p[0] for p in recent_positions) / len(recent_positions)
    avg_y = sum(p[1] for p in recent_positions) / len(recent_positions)
    
    # 检查当前位置与平均位置的距离
    distance = np.sqrt((current_pos[0] - avg_x)**2 + (current_pos[1] - avg_y)**2)
    return distance < STILL_THRESHOLD

def update_screenshot_state(image, hand_sign_id, index_finger_pos):
    """更新截图状态机"""
    global screenshot_state, still_start_time, frame_point1, frame_point2, screenshot_image, current_frame_for_screenshot
    
    current_time = time.time()
    
    # 只在需要截图时才保存帧,减少内存拷贝
    if screenshot_state == STATE_WAITING_CAPTURE:
        current_frame_for_screenshot = image.copy()
    
    if screenshot_state == STATE_IDLE:
        # 空闲状态,等待食指伸出
        if hand_sign_id == 2:  # Point gesture (食指伸出)
            screenshot_state = STATE_DETECTING
            still_start_time = None
            index_finger_history.clear()
    
    elif screenshot_state == STATE_DETECTING:
        # 检测中,等待静止2秒
        if hand_sign_id != 2:
            # 食指收回,重置状态
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
                        # 静止2秒,设置起始点
                        frame_point1 = index_finger_pos
                        frame_point2 = index_finger_pos
                        screenshot_state = STATE_FRAMING
                        still_start_time = None
            else:
                still_start_time = None
    
    elif screenshot_state == STATE_FRAMING:
        # 框定中,跟踪食指移动
        if hand_sign_id != 2:
            # 食指收回,重置状态
            screenshot_state = STATE_IDLE
            still_start_time = None
            frame_point1 = None
            frame_point2 = None
            index_finger_history.clear()
        else:
            # 更新第二个点
            frame_point2 = index_finger_pos
            index_finger_history.append(index_finger_pos)
            
            if is_finger_still(index_finger_pos):
                if still_start_time is None:
                    still_start_time = current_time
                else:
                    elapsed = current_time - still_start_time
                    if elapsed >= STILL_DURATION:
                        # 静止2秒,执行截图
                        screenshot_state = STATE_WAITING_CAPTURE
            else:
                still_start_time = None
    
    elif screenshot_state == STATE_WAITING_CAPTURE:
        # 执行截图
        if frame_point1 and frame_point2:
            screenshot_image = capture_screenshot(current_frame_for_screenshot, frame_point1, frame_point2)
            screenshot_state = STATE_CAPTURED
            # 触发后台上传，避免阻塞视频流
            if last_screenshot_path:
                threading.Thread(target=auto_upload_task, args=(last_screenshot_path,), daemon=True).start()
        else:
            screenshot_state = STATE_IDLE

def capture_screenshot(image, pt1, pt2):
    """执行截图,裁剪指定区域"""
    global last_screenshot_path
    # 确保坐标顺序正确
    x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
    x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
    
    # 确保坐标在图像范围内
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # 裁剪图像
    cropped = image[y1:y2, x1:x2]
    
    # 保存截图
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"screenshot_{timestamp}.jpg"
    filepath = os.path.join(UPLOAD_DIR, filename)
    cv.imwrite(filepath, cropped)
    last_screenshot_path = filepath
    
    print(f"截图已保存: {filename}")
    return cropped

def draw_landmarks(image, landmark_point):
    """在图像上绘制手部关键点和骨架 (参考 app.py 逻辑)"""
    if len(landmark_point) > 0:
        # 连线逻辑
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

    # 性能优化：缩放图像进行 AI 推理 (MediaPipe 使用归一化坐标，不影响原始图像绘图精度)
    h, w = image_rgb.shape[:2]
    image_small = cv.resize(image_rgb, (480, int(h * (480 / w))), interpolation=cv.INTER_AREA)
    image_small.flags.writeable = False
    results = hands.process(image_small)
    
    hand_sign_id = -1  # -1表示无手势, 2表示食指伸出
    index_finger_pos = None
    landmarks_list = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y])

            # 手势识别逻辑
            # 1. OK手势检测: 拇指尖(4)和食指尖(8)距离很近
            dist_4_8 = np.sqrt((landmarks[4][0]-landmarks[8][0])**2 + (landmarks[4][1]-landmarks[8][1])**2)
            
            if dist_4_8 < 0.05:
                hand_sign_id = 3  # OK手势
            elif landmarks[8][1] < landmarks[5][1]:  # 食指伸出
                hand_sign_id = 2  # Pointer手势
            else:
                hand_sign_id = -1
            
            # 获取食指尖端位置 (landmark 8)
            img_width, img_height = image.shape[1], image.shape[0]
            index_finger_pos = [
                int(landmarks[8][0] * img_width),
                int(landmarks[8][1] * img_height)
            ]
            
            landmarks_list = landmarks
            break  # 只处理第一只手
    
    # 更新截图状态机
    update_screenshot_state(image, hand_sign_id, index_finger_pos)
    
    # 只在有手势时绘制关键点
    if landmarks_list is not None:
        image = draw_landmarks(image, landmarks_list)
    
    # 绘制截图框定区域 (仅在检测或框定过程中显示，截图触发后立即移除)
    if frame_point1 and frame_point2 and screenshot_state in [STATE_DETECTING, STATE_FRAMING]:
        cv.rectangle(image, tuple(frame_point1), tuple(frame_point2), (0, 255, 0), 3)
        cv.circle(image, tuple(frame_point1), 8, (255, 0, 0), -1)
    
    # 更新全局手势变量
    if hand_sign_id == 3:
        current_gesture = "ok"
    elif hand_sign_id == 2:
        current_gesture = "pointer"
    else:
        current_gesture = "no hand"
        
    return current_gesture, image

@app.get("/video_feed")
async def video_feed():
    """提供给前端显示的视频流接口"""
    global latest_frame
    if latest_frame is None:
        # 返回空内容并保持 200 状态，避免前端和控制台报 404 错误
        return Response(content=b"", status_code=200, media_type="image/jpeg")
    return Response(content=latest_frame, media_type="image/jpeg")

@app.get("/screenshot_status")
async def screenshot_status():
    """获取当前截图状态"""
    global screenshot_state, frame_point1, frame_point2, current_gesture
    
    return JSONResponse(content={
        "state": screenshot_state,
        "point1": frame_point1,
        "point2": frame_point2,
        "gesture": current_gesture,
        "has_screenshot": screenshot_image is not None
    })

@app.get("/get_screenshot")
async def get_screenshot():
    """获取截图结果"""
    global screenshot_image
    
    if screenshot_image is None:
        return Response(content=b"", status_code=404)
    
    # 将截图编码为JPEG
    _, buffer = cv.imencode('.jpg', screenshot_image)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

def reset_screenshot_logic():
    """内部重置逻辑：清空状态、坐标历史和截图结果"""
    global screenshot_state, still_start_time, frame_point1, frame_point2, screenshot_image, latest_ai_response, last_screenshot_path, current_gesture, current_frame_for_screenshot
    
    screenshot_state = STATE_IDLE
    still_start_time = None
    frame_point1 = None
    frame_point2 = None
    screenshot_image = None
    current_frame_for_screenshot = None
    current_gesture = "no hand"
    latest_ai_response = ""
    last_screenshot_path = None
    index_finger_history.clear()

@app.post("/reset_screenshot")
async def reset_screenshot():
    """重置截图状态接口"""
    reset_screenshot_logic()
    
    return JSONResponse(content={"message": "Screenshot state reset"})

@app.get("/get_ai_response")
async def get_ai_response():
    """获取最新的 AI 分析结果"""
    global latest_ai_response
    return JSONResponse(content={"analysis": latest_ai_response or ""})

@app.post("/toggle_rotate")
async def toggle_rotate():
    """切换旋转角度 (0, 90, 180, 270)"""
    global rotation_angle
    rotation_angle = (rotation_angle + 90) % 360
    reset_screenshot_logic()  # 旋转后必须重置坐标系
    return JSONResponse(content={"angle": rotation_angle})

@app.post("/toggle_mirror")
async def toggle_mirror():
    """切换镜像状态"""
    global is_mirror
    is_mirror = not is_mirror
    reset_screenshot_logic()  # 镜像后必须重置坐标系
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

def usb_camera_loop():
    """USB摄像头处理线程"""
    global latest_frame, camera_running
    
    print("尝试打开USB摄像头...")
    # 1. 尝试使用 DirectShow 后端 (Windows 推荐，启动更快且更稳定)
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    
    # 2. 如果失败，尝试默认后端
    if not cap.isOpened():
        cap = cv.VideoCapture(0)
    
    # 3. 如果索引 0 仍失败，尝试索引 1 (部分系统会将虚拟摄像头或集成摄像头设为 0)
    if not cap.isOpened():
        cap = cv.VideoCapture(1)

    if not cap.isOpened():
        print("错误: 无法打开任何USB摄像头。请检查：\n1. 摄像头是否已插入\n2. 隐私设置中是否允许应用访问摄像头\n3. 是否有其他程序（如微信、Zoom）正在占用摄像头")
        camera_running = False
        return

    # 设置分辨率为高清 (1280x720)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    print("USB摄像头已打开")

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
            
        # 应用旋转和镜像
        if rotation_angle == 90:
            frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            frame = cv.rotate(frame, cv.ROTATE_180)
        elif rotation_angle == 270:
            frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
        
        if is_mirror:
            frame = cv.flip(frame, 1)
        frame = frame.copy()
        
        # 手势识别处理
        try:
            _, debug_image = classify_gesture(frame)
            
            # 性能优化：降低预览流编码质量，减少 CPU 占用
            encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 80]
            latest_frame = cv.imencode('.jpg', debug_image, encode_param)[1].tobytes()
        except Exception as e:
            print(f"处理帧出错: {e}")
        
        # 减小人为延迟
        time.sleep(0.01)
        
    cap.release()
    print("USB摄像头已释放")

if __name__ == "__main__":
    import uvicorn
    # access_log=False 禁用海量的访问日志，让控制台只显示关键业务信息
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
