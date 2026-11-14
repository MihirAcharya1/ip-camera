import os
import time
import datetime
import json
import cv2
import torch
from ultralytics import YOLO

# 1. Hardware & Network Tuning Flags
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|tls_verify;0"

# Target the NVIDIA Ampere GPU on your Jetson Orin Nano
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[+] Directing Framework Target To: {device.upper()}")

username = "user"
password = "india$Me123"
ip_address = "192.168.31.241"
port = "554"
secure_rtsp_url = f"rtsps://{username}:{password}@{ip_address}:{port}/video/live?channel=1&subtype=1"

JSON_LOG_FILE = "session_log.json"
JSON_WRITE_INTERVAL = 2.0  # Throttled I/O updates to maximize edge performance
last_json_write_time = 0.0

def save_database_to_json(db, start_time):
    """Encapsulates tracking states into structured JSON format on disk."""
    total_elapsed = time.time() - start_time
    output_data = {
        "session_meta": {
            "start_time": datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "current_update_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_runtime_seconds": round(total_elapsed, 2)
        },
        "tracked_people": {}
    }
    for pid, data in db.items():
        prod_score = (data["total_time_inside"] / total_elapsed) * 100 if total_elapsed > 0 else 0
        output_data["tracked_people"][str(pid)] = {
            "first_detected": datetime.datetime.fromtimestamp(data["first_seen"]).strftime("%H:%M:%S"),
            "total_time_present_seconds": round(data["total_time_inside"], 2),
            "currently_in_frame": data["is_present"],
            "productivity_rating_percentage": round(prod_score, 1)
        }
    with open(JSON_LOG_FILE, "w") as f:
        json.dump(output_data, f, indent=4)

# 2. Native Initialization using TensorRT Engine
if os.path.exists("yolo11n.engine"):
    print("[🚀 TENSORRT DETECTED] Loading compiled hardware engine map...")
    model = YOLO("yolo11n.engine")
else:
    print("[⚠️ WARNING] Running raw PyTorch. Expecting severe CPU drops.")
    model = YOLO("yolo11n.pt")

model.to(device)

# Clear Tensor Core startup memory lag with a dummy frame warmup pass
print("[+] Running hardware execution pipeline warmup...")
dummy_frame = torch.zeros((1, 3, 640, 640), device=device)
_ = model(dummy_frame, verbose=False)

print(f"[+] Interfacing Secure Camera Network Feed: {secure_rtsp_url}")
cap = cv2.VideoCapture(secure_rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    secure_rtsp_url_alt = f"rtsps://{username}:{password}@{ip_address}:{port}/cam/realmonitor?channel=1&subtype=1"
    cap = cv2.VideoCapture(secure_rtsp_url_alt, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("[-] Error: Camera connection dropped at firewall layer.")
    exit()

# Set internal frame buffers to 1 to minimize delay buildup
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

person_database = {}
SHIFT_START_TIME = time.time()
window_name = "Jetson Orin Nano AI Tracker"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

print("\n[+] Tracking Operational. Real-time updates logging to session_log.json")

try:
    while True:
        # Flush the buffer stack to always capture the current live frame
        for _ in range(2): 
            cap.grab()
        ret, frame = cap.retrieve()
        
        if not ret:
            print("Dropped video packet frames.")
            break

        current_time = time.time()
        total_elapsed_shift = current_time - SHIFT_START_TIME

        # Run high-performance tracking (Class 0: Person)
        results = model.track(frame, persist=True, classes=[0], verbose=False, device=device)
        active_ids_this_frame = set()

        if results and results.boxes.id is not None:
            boxes = results.boxes.xyxy.int().cpu().tolist()
            track_ids = results.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                active_ids_this_frame.add(track_id)

                if track_id not in person_database:
                    enter_time = datetime.datetime.now().strftime("%H:%M:%S")
                    print(f"\n[➔ ENTER] Person ID {track_id} entered at {enter_time}")
                    person_database[track_id] = {
                        "first_seen": current_time,
                        "last_frame_time": current_time,
                        "total_time_inside": 0.0,
                        "is_present": True
                    }
                else:
                    if not person_database[track_id]["is_present"]:
                        re_enter_time = datetime.datetime.now().strftime("%H:%M:%S")
                        print(f"\n[➔ RE-ENTER] Person ID {track_id} returned at {re_enter_time}")
                        person_database[track_id]["is_present"] = True
                        person_database[track_id]["last_frame_time"] = current_time
                    
                    time_delta = current_time - person_database[track_id]["last_frame_time"]
                    person_database[track_id]["total_time_inside"] += time_delta
                    person_database[track_id]["last_frame_time"] = current_time

                time_inside = person_database[track_id]["total_time_inside"]
                productivity_score = (time_inside / total_elapsed_shift) * 100 if total_elapsed_shift > 0 else 0

                # Render Bounding Box HUD
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"ID: {track_id} | Prod: {productivity_score:.1f}%"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Handle Absence Calculations
        for pid, data in person_database.items():
            if data["is_present"] and pid not in active_ids_this_frame:
                if current_time - data["last_frame_time"] > 2.0:
                    data["is_present"] = False
                    leave_time = datetime.datetime.now().strftime("%H:%M:%S")
                    print(f"[➔ LEAVE] Person ID {pid} exited at {leave_time}")
        
        # Throttled live update checkpoint
        if current_time - last_json_write_time >= JSON_WRITE_INTERVAL:
            save_database_to_json(person_database, SHIFT_START_TIME)
            last_json_write_time = current_time

        cv2.imshow(window_name, frame)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("\n[+] Finalizing data sync before closure...")
    save_database_to_json(person_database, SHIFT_START_TIME)
    cap.release()
    cv2.destroyAllWindows()
    print(f"[+] Mission Complete. Final history saved securely to '{JSON_LOG_FILE}'.")
