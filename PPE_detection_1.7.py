import os
import cv2
import yaml
import torch
import threading
import subprocess
import numpy as np
import 
from time import time
from datetime import datetime
from collections import deque, defaultdict
from scipy.spatial import distance
from ultralytics import YOLO
from violation_flow import handle_violation  

# CONFIGURATION
INFER_SIZE = (960, 600)
STREAM_SIZE = (960, 600)
distance_threshold = 500
infer_every = 5
MAX_MISSING_FRAMES = 5
violation_cooldowns = {}
VIOLATION_COOLDOWN_SEC = 30

THIS_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(THIS_DIR, "config.json")
CONFIG_YAML_PATH = os.path.join(THIS_DIR, "config.yaml")
DETECTIONS_DIR = os.path.join(THIS_DIR, "detections")
CSV_LOG = None
LOG_FIELDS = []
with open(CONFIG_YAML_PATH, "r") as f:
    config_sett = yaml.safe_load(f)

MODEL_PATH = os.path.join(THIS_DIR, "11m100.engine")

assert os.path.exists(CONFIG_PATH), "Missing config.json"
assert os.path.exists(MODEL_PATH), "Missing model.engine"

# GLOBAL STATE
missing_frame_counts = defaultdict(int)
violated_ids = set()
violation_cooldowns = {}  # PID → timestamp

person_tracks = {}
next_person_id = 0
status_history = defaultdict(lambda: deque(maxlen=10))
previous_centers = {}
people_entered = 0
people_exited = 0
people_inside = 0

# SHARED BUFFER FOR STREAMING
class LatestFrame:
    buffer = None
    lock = threading.Lock()

    @classmethod
    def set(cls, frame):
        with cls.lock:
            cls.buffer = frame.copy()

    @classmethod
    def get(cls):
        with cls.lock:
            return None if cls.buffer is None else cls.buffer.copy()

# LOAD CONFIG
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

CONF_THRESHOLD = config["detection_logic"].get("min_confidence", 0.7)
CSV_LOG = os.path.join(THIS_DIR, config["log_behavior"]["csv_path"])
LOG_FIELDS = config["log_behavior"]["fields"]
os.makedirs(os.path.dirname(CSV_LOG), exist_ok=True)
if not os.path.exists(CSV_LOG):
    with open(CSV_LOG, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()
os.makedirs(DETECTIONS_DIR, exist_ok=True)
PPE_CLASSES = {
    "helmet_OK": 0, "helmet_NOT": 2,
    "mask_OK": 1, "mask_NOT": 3,
    "vest_OK": 7, "vest_NOT": 4,
}

# LOAD MODEL
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = YOLO(MODEL_PATH)

def apply_camera_settings(device="/dev/video0"):
    settings = config_sett.get("camera", {}).get("settings", [])
    for setting in settings:
        print(f"📷 Applying camera setting: {setting}")
        subprocess.run(f"v4l2-ctl -d {device} -c {setting}", shell=True, check=False)
#apply_camera_settings()        
        
def overlaps(boxA, boxB, threshold=0.1):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    areaA = max(1, (ax2 - ax1) * (ay2 - ay1))
    return (inter_area / areaA) > threshold

def assign_person_ids(current_boxes, person_tracks, next_id, available_ids):
    assigned = {}
    for pid, track in person_tracks.items():
        if not track:
            continue
        last_box = track[-1]
        lx = (last_box[0] + last_box[2]) / 2
        ly = (last_box[1] + last_box[3]) / 2
        best_idx, best_dist = -1, float("inf")
        for i, box in enumerate(current_boxes):
            if i in assigned:
                continue
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            d = distance.euclidean((lx, ly), (cx, cy))
            if d < best_dist and d < distance_threshold:
                best_dist = d
                best_idx = i
        if best_idx != -1:
            assigned[best_idx] = pid
            person_tracks[pid].append(current_boxes[best_idx])

    for i, box in enumerate(current_boxes):
        if i not in assigned:
            reuse_id = available_ids.pop() if available_ids else next_id
            person_tracks[reuse_id] = deque([box], maxlen=5)
            assigned[i] = reuse_id
            if reuse_id == next_id:
                next_id += 1
    return assigned, person_tracks, next_id, available_ids

def detection_loop():
    global next_person_id, person_tracks, people_entered, people_exited, people_inside

    frame_count = 0
    available_ids = set()
    last_annotated = None

    pipeline = (
        f"v4l2src device=/dev/video0 ! "
        f"image/jpeg, width=INFER_SIZE[0], height=INFER_SIZE[1], framerate=30/1 ! "
        "nvv4l2decoder mjpeg=1 ! "
        "nvvidconv ! "
        "videoconvert ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    assert cap.isOpened(), "❌ Failed to open camera"

    print("📷 detection_loop started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % infer_every != 0:
            LatestFrame.set(last_annotated if last_annotated is not None else cv2.resize(frame, STREAM_SIZE))
            continue

        resized = cv2.resize(frame, INFER_SIZE)
        results = model.predict(source=resized, conf=CONF_THRESHOLD, verbose=False)

        people, items = [], []
        detections_for_logging = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                conf_v = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                detections_for_logging.append((cls_id, conf_v, x1, y1, x2, y2))
                if cls_id == 5:
                    people.append((x1, y1, x2, y2))
                elif cls_id in PPE_CLASSES.values():
                    items.append((x1, y1, x2, y2, cls_id))

        assigned, person_tracks, next_person_id, available_ids = assign_person_ids(people, person_tracks, next_person_id, available_ids)
        active_persons = {assigned[i]: box for i, box in enumerate(people)}

        for pid in list(status_history.keys()):
            if pid not in active_persons:
                missing_frame_counts[pid] += 1
                if missing_frame_counts[pid] <= MAX_MISSING_FRAMES:
                    if pid in person_tracks and person_tracks[pid]:
                        active_persons[pid] = person_tracks[pid][-1]
                else:
                    status_history.pop(pid, None)
                    missing_frame_counts.pop(pid, None)
                    person_tracks.pop(pid, None)
                    available_ids.add(pid)
                    violated_ids.discard(pid)
                    previous_centers.pop(pid, None)
                    violation_cooldowns.pop(pid, None)
            else:
                missing_frame_counts[pid] = 0

        annotated = resized.copy()
        violation_found = False
        violation_label = None
        violation_conf = 0.0
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for pid, (x1, y1, x2, y2) in active_persons.items():
            head = (x1, y1, x2, y1 + (y2 - y1) // 3)
            torso = (x1, y1 + (y2 - y1) // 3, x2, y1 + 2 * (y2 - y1) // 3)
            statuses = {"helmet": False, "mask": False, "vest": False}

            for (ix1, iy1, ix2, iy2, cls_j) in items:
                if cls_j == PPE_CLASSES["helmet_OK"] and overlaps((ix1, iy1, ix2, iy2), head):
                    statuses["helmet"] = True
                elif cls_j == PPE_CLASSES["mask_OK"] and overlaps((ix1, iy1, ix2, iy2), head):
                    statuses["mask"] = True
                elif cls_j == PPE_CLASSES["vest_OK"] and overlaps((ix1, iy1, ix2, iy2), torso):
                    statuses["vest"] = True

            status_history[pid].append(statuses)
            window = list(status_history[pid])
            smoothed = {k: sum(hist[k] for hist in window) >= (len(window) // 2 + 1) for k in statuses}

            # Entry/exit
            center_x = (x1 + x2) // 2
            if pid in previous_centers:
                prev_x = previous_centers[pid]
                if prev_x < INFER_SIZE[0] // 2 <= center_x:
                    people_entered += 1
                    people_inside += 1
                elif prev_x > INFER_SIZE[0] // 2 >= center_x:
                    people_exited += 1
                    people_inside = max(0, people_inside - 1)
            previous_centers[pid] = center_x
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated, f"ID {pid}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Violation check + cooldown
            now = time()
            if not all(smoothed.values()):
                if now - violation_cooldowns.get(pid, 0) >= VIOLATION_COOLDOWN_SEC:
                    violation_cooldowns[pid] = now
                    violated_ids.add(pid)
                    violation_found = True
                    missing = [k for k, v in smoothed.items() if not v]
                    violation_label = "_".join(missing)
                    for (c_k, conf_k, *_ ) in detections_for_logging:
                        nm = model.names[c_k].lower()
                        if nm in missing and conf_k > violation_conf:
                            violation_conf = conf_k
                          
                '''if pid not in violated_ids:
                    # First print this violation episode
                    violated_ids.add(pid)
                    violation_cooldowns[pid] = now
                    print(f"🚨 Violation: ID={pid} at {datetime.now().strftime('%H:%M:%S')}")
                else:
                    # If cooldown expired, print again
                    last_print = violation_cooldowns.get(pid, 0)
                    if now - last_print >= VIOLATION_COOLDOWN_SEC:
                        violation_cooldowns[pid] = now
                        print(f"🚨 Violation (cooldown): ID={pid} at {datetime.now().strftime('%H:%M:%S')}")'''
            else:
                # Now compliant—reset everything for next episode
                violated_ids.discard(pid)
                violation_cooldowns.pop(pid, None)                 
            x_text = x2 - 120
            y_text = y1 + 20
            labels = [
                ("Helmet OK" if smoothed["helmet"] else "NO HELMET", (60, 190, 20) if smoothed["helmet"] else (30, 50, 230)),
                ("Mask OK" if smoothed["mask"] else "NO MASK", (60, 190, 20) if smoothed["mask"] else (30, 50, 230)),
                ("Vest OK" if smoothed["vest"] else "NO VEST", (60, 190, 20) if smoothed["vest"] else (30, 50, 230)),
            ]
            for text, color in labels:
                cv2.putText(annotated, text, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_text += 25
        summary_text = f"In Frame: {len(active_persons)}  Entered: {people_entered}  Exited: {people_exited}  Inside: {people_inside}"
        cv2.putText(annotated, summary_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 255), 2)

        if violation_found and violation_label:
            fname = f"{violation_label}_{ts}.jpg"
            cv2.imwrite(os.path.join(DETECTIONS_DIR, fname), annotated)
            with open(CSV_LOG, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow({
                    "timestamp": ts,
                    "label": violation_label,
                    "confidence": f"{violation_conf:.2f}",
                    "status": "violation"
                })
            metrics = {
                "total_entry_count": {"label": "Total persons entered", "value": people_entered},
                "total_exit_count": {"label": "Total persons exited", "value": people_exited},
                "current_frame_count": {"label": "People in frame", "value": len(active_persons)},
                "total_in_count" : { "label" : "Total people in zone" , "value" : people_inside}
            }
            threading.Thread(
                target=handle_violation,
                args=(violation_label, violation_conf, annotated.copy(), metrics),
                daemon=True
            ).start()

        last_annotated = cv2.resize(annotated, STREAM_SIZE)
        LatestFrame.set(last_annotated)

# MAIN
if __name__ == "__main__":
    threading.Thread(target=detection_loop, daemon=True).start()
    print("✅ detection_loop started.")
    while True:
        cv2.waitKey(1000)
