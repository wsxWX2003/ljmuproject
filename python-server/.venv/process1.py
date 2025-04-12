import cv2
import numpy as np
import torch
import sqlite3
import requests
from flask import Flask, Response, jsonify
from threading import Thread, Lock
from queue import Queue
import sys
import time
import os
from datetime import datetime
from PIL import Image
import imagehash

# 添加自定义路径
sys.path.append(r"F:\LjmuStudy\engineeringProject\ObjectDetect\sort-master")
from sort import Sort

sys.path.append(r"F:\LjmuStudy\engineeringProject\ObjectDetect\yolov5-5.0")
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

ESP32_CAM_URL = "http://192.168.4.1:80/stream"
MODEL_PATH = r"F:\LjmuStudy\engineeringProject\ObjectDetect\yolov5-5.0\runs\train\exp5\weights\best.pt"
DB_PATH = r'F:\LjmuStudy\engineeringProject\DataBase\yolo.db'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load(MODEL_PATH, map_location=device)
model.eval()
names = model.module.names if hasattr(model, 'module') else model.names

app = Flask(__name__)
tracker = Sort()
lock = Lock()
tracked_ids = set()
garbage_count = {}
frame_queue = Queue(maxsize=1)

conf_threshold = 0.75
iou_threshold = 0.45
min_box_area = 1000
blur_threshold = 100.0

os.makedirs("captures", exist_ok=True)
saved_hashes = []

# ✅ 修改后的建表（不使用 AUTOINCREMENT）
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS GarbageStats (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    count INTEGER NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()

# ✅ 修改后的插入函数：手动控制 id 累计
def insert_garbage_stats(garbage_type, count):
    cursor.execute('SELECT MAX(id) FROM GarbageStats')
    result = cursor.fetchone()
    next_id = (result[0] or 0) + 1
    cursor.execute('INSERT INTO GarbageStats (id, type, count) VALUES (?, ?, ?)', (next_id, garbage_type, count))
    conn.commit()

def detect_objects(img):
    img_resized = letterbox(img, new_shape=640)[0]
    img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)
    img_resized = np.ascontiguousarray(img_resized)
    img_tensor = torch.from_numpy(img_resized).float().to(device) / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, conf_threshold, iou_threshold)[0]
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img.shape).round()
    return pred

def is_new_object(roi):
    if roi.size == 0:
        return False
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        if fm < blur_threshold:
            print("⛔ 模糊图像，跳过保存")
            return False
        h = imagehash.phash(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)))
        for old in saved_hashes:
            if abs(h - old) < 5:
                print("⚠️ 排重：图像重复")
                return False
        saved_hashes.append(h)
        return True
    except Exception as e:
        print("哈希计算失败:", e)
        return False

def mjpeg_stream_reader():
    while True:
        try:
            stream = requests.get(ESP32_CAM_URL, stream=True, timeout=5)
            bytes_data = bytes()
            for chunk in stream.iter_content(chunk_size=1024):
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        if not frame_queue.empty():
                            frame_queue.get_nowait()
                        frame_queue.put(img)
        except Exception as e:
            print("ESP32连接失败，重试中...", e)
            time.sleep(2)

Thread(target=mjpeg_stream_reader, daemon=True).start()

def generate_frames():
    frame_id = 0
    pred = None
    while True:
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()
        if frame_id % 2 == 0:
            pred = detect_objects(frame)

        if pred is not None and len(pred):
            dets = np.array([[*det[:4].cpu().numpy(), det[4].item(), det[5].item()] for det in pred])
            tracked_objs = tracker.update(dets[:, :5])

            for i, obj in enumerate(tracked_objs):
                x1, y1, x2, y2, tid = obj.astype(int)
                cls = int(dets[i][5])
                class_name = names[cls]

                # ✅ 只显示 bottle 和 plastic_bag，can 映射为 bottle
                if class_name == "can":
                    class_name = "bottle"
                elif class_name not in ["bottle", "plastic"]:
                    continue

                label = f"{class_name} ID:{tid}"
                area = (x2 - x1) * (y2 - y1)
                ratio = (x2 - x1) / (y2 - y1 + 1e-6)

                if area < min_box_area or ratio < 0.3 or ratio > 3.5:
                    continue
                if tid not in tracked_ids:
                    tracked_ids.add(tid)
                    with lock:
                        garbage_count[class_name] = garbage_count.get(class_name, 0) + 1
                    insert_garbage_stats(class_name, 1)

                    roi = frame[y1:y2, x1:x2]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if is_new_object(roi):
                        filename = f"captures/{class_name}_{tid}_{timestamp}.jpg"
                        cv2.imwrite(filename, roi)
                        print(f"✅ 保存截图: {filename}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_id += 1
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/garbage_stats')
def stats():
    with lock:
        return jsonify(garbage_count)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
