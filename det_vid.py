import sys
import os
import cv2
import yaml
import csv
import threading
import subprocess
import configparser
import numpy as np
from time import time
from datetime import datetime
from collections import deque, defaultdict
from scipy.spatial import distance

sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from is_aarch64 import is_aarch64
from bus_call import bus_call
import pyds

# Import your violation handling
from violation_flow import handle_violation

# CONFIGURATION
THIS_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(THIS_DIR, "config.json")
CONFIG_YAML_PATH = os.path.join(THIS_DIR, "config.yaml")
DETECTIONS_DIR = os.path.join(THIS_DIR, "detections")
VIOLATION_VIDEOS_DIR = os.path.join(THIS_DIR, "violation_videos")

# Load configurations
with open(CONFIG_YAML_PATH, "r") as f:
    config_sett = yaml.safe_load(f)

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Constants from your original code
CONF_THRESHOLD = config["detection_logic"]["min_confidence"]
CSV_LOG = os.path.join(THIS_DIR, config["log_behavior"]["csv_path"])
LOG_FIELDS = config["log_behavior"]["fields"]
VIOLATION_COOLDOWN_SEC = 30

# Video recording constants
VIDEO_BUFFER_SECONDS = 30  # 30 seconds before and after violation
VIDEO_FPS = 30  # Adjust based on your camera FPS
VIDEO_BUFFER_SIZE = VIDEO_BUFFER_SECONDS * VIDEO_FPS

# Ensure directories exist
os.makedirs(os.path.dirname(CSV_LOG), exist_ok=True)
os.makedirs(DETECTIONS_DIR, exist_ok=True)
os.makedirs(VIOLATION_VIDEOS_DIR, exist_ok=True)

# Initialize CSV if it doesn't exist
if not os.path.exists(CSV_LOG):
    with open(CSV_LOG, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

# CLASS NAMES - Updated to match your detection classes
CLASS_NAMES = {
    0: "person", 1: "ear", 2: "ear-mufs", 3: "face", 4: "face-guard", 5: "face-mask",
    6: "foot", 7: "tool", 8: "glasses", 9: "gloves", 10: "helmet", 11: "hands",
    12: "head", 13: "medical-suit", 14: "shoes", 15: "safety-suit", 16: "safety-vest"
}

PPE_CLASSES = {
    "helmet": 10,
    "mask": 5,
    "vest": 16,
    "gloves": 9
}

# Global tracking variables
seen_ids_by_class = {cls_id: set() for cls_id in CLASS_NAMES}
status_history = defaultdict(lambda: deque(maxlen=10))
violation_cooldowns = {}
violated_ids = set()
previous_centers = {}
person_violation_recorded = set()  # Track which persons already have recorded violations

# People counting variables
people_entered = 0
people_exited = 0
people_inside = 0
people_entered_with_violation = 0
people_entered_with_full_ppe = 0

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

class VideoBuffer:
    """Circular buffer to store video frames for violation recording"""
    def __init__(self, max_size=VIDEO_BUFFER_SIZE):
        self.frames = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.max_size = max_size
    
    def add_frame(self, frame, timestamp):
        with self.lock:
            self.frames.append(frame.copy())
            self.timestamps.append(timestamp)
    
    def get_frames_around_time(self, violation_time, before_sec=30, after_sec=30):
        """Get frames from before_sec seconds before to after_sec seconds after violation_time"""
        with self.lock:
            if not self.frames:
                return []
            
            # Find the index closest to violation time
            violation_idx = 0
            min_diff = float('inf')
            for i, ts in enumerate(self.timestamps):
                diff = abs(ts - violation_time)
                if diff < min_diff:
                    min_diff = diff
                    violation_idx = i
            
            # Calculate frame range
            frames_before = int(before_sec * VIDEO_FPS)
            frames_after = int(after_sec * VIDEO_FPS)
            
            start_idx = max(0, violation_idx - frames_before)
            end_idx = min(len(self.frames), violation_idx + frames_after)
            
            selected_frames = []
            for i in range(start_idx, end_idx):
                selected_frames.append(self.frames[i].copy())
            
            return selected_frames

# Global video buffer
video_buffer = VideoBuffer()

def save_violation_video(person_id, violation_label, violation_time):
    """Save video with frames around violation time"""
    try:
        # Get frames around violation time
        frames = video_buffer.get_frames_around_time(violation_time)
        
        if not frames:
            print(f"⚠️ No frames available for violation video for person {person_id}")
            return
        
        # Create video filename
        timestamp_str = datetime.fromtimestamp(violation_time).strftime('%Y%m%d_%H%M%S')
        video_filename = f"violation_person_{person_id}_{violation_label}_{timestamp_str}.avi"
        video_path = os.path.join(VIOLATION_VIDEOS_DIR, video_filename)
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (width, height))
        
        if not video_writer.isOpened():
            print(f"❌ Could not open video writer for {video_path}")
            return
        
        # Write frames to video
        for frame in frames:
            video_writer.write(frame)
        
        video_writer.release()
        print(f"✅ Saved violation video: {video_filename} ({len(frames)} frames)")
        
    except Exception as e:
        print(f"❌ Error saving violation video for person {person_id}: {e}")

def overlaps(boxA, boxB, threshold=0.1):
    """Check if two bounding boxes overlap"""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    areaA = max(1, (ax2 - ax1) * (ay2 - ay1))
    return (inter_area / areaA) > threshold

# Frame dimensions (adjust based on your camera)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 800

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """Main buffer probe function that handles detection and tracking"""
    global seen_ids_by_class, status_history, violation_cooldowns, violated_ids
    global previous_centers, people_entered, people_exited, people_inside
    global people_entered_with_violation, people_entered_with_full_ppe, person_violation_recorded
    
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    # Get current frame for video buffer
    current_frame = LatestFrame.get()
    current_time = time()
    if current_frame is not None:
        video_buffer.add_frame(current_frame, current_time)

    while l_frame is not None:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list

        # Collect all detections
        people_detections = {}
        ppe_detections = []
        
        # Count detections for verification
        detection_counts = {"person": 0, "helmet": 0, "mask": 0, "vest": 0, "gloves": 0}
        
        while l_obj is not None:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            class_id = obj_meta.class_id
            object_id = obj_meta.object_id
            confidence = obj_meta.confidence
            
            # Get bounding box coordinates
            left = int(obj_meta.rect_params.left)
            top = int(obj_meta.rect_params.top)
            width = int(obj_meta.rect_params.width)
            height = int(obj_meta.rect_params.height)
            right = left + width
            bottom = top + height
            
            if class_id == 0:  # Person class
                detection_counts["person"] += 1
                
                # Track unique person IDs
                if object_id not in seen_ids_by_class[class_id]:
                    seen_ids_by_class[class_id].add(object_id)
                
                people_detections[object_id] = {
                    'bbox': (left, top, right, bottom),
                    'confidence': confidence,
                    'obj_meta': obj_meta
                }
                
                # Set person bounding box - default blue
                obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 1.0)
                obj_meta.rect_params.border_width = 2
                
                if hasattr(obj_meta, 'text_params'):
                    obj_meta.text_params.display_text = ""

            else:
                # Completely hide all non-person classes
                obj_meta.rect_params.border_width = 0
                obj_meta.rect_params.border_color.set(0.0, 0.0, 0.0, 0.0)
                if hasattr(obj_meta, 'text_params'):
                    obj_meta.text_params.display_text = ""
                    obj_meta.text_params.set_bg_clr = 0
                    obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.0)

                # Also skip adding to PPE list if you don't need logic at all
                # (Optional: Comment this block out if you want to skip PPE logic altogether)
                if class_id in PPE_CLASSES.values():
                    ppe_detections.append({
                        'class_id': class_id,
                        'bbox': (left, top, right, bottom),
                        'confidence': confidence,
                        'obj_meta': obj_meta
                    })
            
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Process PPE detection for each person
        violation_found = False
        violation_label = None
        violation_conf = 0.0
        violation_person_id = None
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for person_id, person_data in people_detections.items():
            person_bbox = person_data['bbox']
            x1, y1, x2, y2 = person_bbox
            
            # Define body regions for PPE detection
            head = (x1, y1, x2, y1 + (y2 - y1) // 3)
            torso = (x1, y1 + (y2 - y1) // 3, x2, y1 + 2 * (y2 - y1) // 3)
            hands_region = (x1, y1 + (y2 - y1) // 2, x2, y2)

            # Check PPE status
            statuses = {"helmet": False, "mask": False, "vest": False, "gloves": False}
            
            for ppe_item in ppe_detections:
                ppe_bbox = ppe_item['bbox']
                ppe_class_id = ppe_item['class_id']
                
                if ppe_class_id == PPE_CLASSES["helmet"] and overlaps(ppe_bbox, head):
                    statuses["helmet"] = True
                elif ppe_class_id == PPE_CLASSES["mask"] and overlaps(ppe_bbox, head):
                    statuses["mask"] = True
                elif ppe_class_id == PPE_CLASSES["vest"] and overlaps(ppe_bbox, torso):
                    statuses["vest"] = True
                elif ppe_class_id == PPE_CLASSES["gloves"] and overlaps(ppe_bbox, hands_region):
                    statuses["gloves"] = True

            # Update status history
            status_history[person_id].append(statuses)
            window = list(status_history[person_id])
            smoothed = {k: sum(hist[k] for hist in window) >= (len(window) // 2 + 1) for k in statuses}

            # Track entry/exit based on center position
            center_x = (x1 + x2) // 2
            if person_id in previous_centers:
                prev_x = previous_centers[person_id]
                if prev_x < FRAME_WIDTH // 2 <= center_x:
                    people_entered += 1
                    people_inside += 1
                    if not all(smoothed.values()):
                        people_entered_with_violation += 1
                    else:
                        people_entered_with_full_ppe += 1
                elif prev_x > FRAME_WIDTH // 2 >= center_x:
                    people_exited += 1
                    people_inside = max(0, people_inside - 1)
            previous_centers[person_id] = center_x

            # Check for violations and set box color
            now = time()
            if not all(smoothed.values()):
                # Only record violation if this person hasn't been recorded yet
                if (person_id not in person_violation_recorded and 
                    now - violation_cooldowns.get(person_id, 0) >= VIOLATION_COOLDOWN_SEC):
                    
                    violation_cooldowns[person_id] = now
                    violated_ids.add(person_id)
                    person_violation_recorded.add(person_id)  # Mark this person as recorded
                    violation_found = True
                    violation_person_id = person_id
                    missing = [k for k, v in smoothed.items() if not v]
                    violation_label = "_".join(missing)
                    
                    # Find confidence for violated items
                    for ppe_item in ppe_detections:
                        ppe_class_name = CLASS_NAMES.get(ppe_item['class_id'], "").lower()
                        if ppe_class_name in missing and ppe_item['confidence'] > violation_conf:
                            violation_conf = ppe_item['confidence']
                
                # Set person bounding box color to red for violation
                person_data['obj_meta'].rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)
                person_data['obj_meta'].rect_params.border_width = 3
            else:
                # Set person bounding box color to green for compliant
                person_data['obj_meta'].rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)
                person_data['obj_meta'].rect_params.border_width = 2
                violated_ids.discard(person_id)
                # Don't reset violation_cooldowns or person_violation_recorded to prevent duplicates

            # Add PPE status text to person box
            obj_meta = person_data['obj_meta']
            ppe_status_text = []
            for ppe_type, status in smoothed.items():
                status_symbol = "✓" if status else "✗"
                ppe_status_text.append(f"{ppe_type[0].upper()}{ppe_type[1:]}: {status_symbol}")
            
            # Set text overlay for this person
            if hasattr(obj_meta, 'text_params'):
                violation_status = " [RECORDED]" if person_id in person_violation_recorded else ""
                obj_meta.text_params.display_text = f"ID:{person_id}{violation_status} | " + " | ".join(ppe_status_text)
                obj_meta.text_params.x_offset = int(obj_meta.rect_params.left)
                obj_meta.text_params.y_offset = max(20, int(obj_meta.rect_params.top) - 35)
                obj_meta.text_params.font_params.font_name = "Serif"
                obj_meta.text_params.font_params.font_size = 10
                obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                obj_meta.text_params.set_bg_clr = 1
                obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.8)

        # Create display overlays
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        
        # Top-left stats panel
        stats_lines = [
            f"Frame: {frame_number}",
            f"People: {len(people_detections)}",
            f"Entered: {people_entered}",
            f"Exited: {people_exited}",
            f"Inside: {people_inside}",
            f"Violations: {people_entered_with_violation}",
            f"Valid PPE: {people_entered_with_full_ppe}",
            f"Recorded: {len(person_violation_recorded)}"
        ]
        
        # Stats panel in top-left
        stats_summary = " | ".join(stats_lines)
        display_meta.num_labels = 1  # Only 1 label for single-line stats

        text_params = display_meta.text_params[0]
        text_params.display_text = stats_summary
        text_params.x_offset = 10
        text_params.y_offset = 10
        text_params.font_params.font_name = "Serif"
        text_params.font_params.font_size = 11
        text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)  # White
        text_params.set_bg_clr = 1
        text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.9)  # Black background

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        # Handle violations (existing code + video recording)
        if violation_found and violation_label and violation_person_id:
            fname = f"{violation_label}_{ts}.jpg"
            
            with open(CSV_LOG, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow({
                    "timestamp": ts,
                    "label": violation_label,
                    "confidence": f"{violation_conf:.2f}",
                    "status": "violation",
                    "person_id": violation_person_id
                })
            
            metrics = {
                "total_entry_count": {"label": "Total persons entered", "value": people_entered},
                "total_exit_count": {"label": "Total persons exited", "value": people_exited},
                "current_frame_count": {"label": "People in frame", "value": len(people_detections)},
                "total_in_count": {"label": "Total people in zone", "value": people_inside},
                "without_ppe_count": {"label": "Violation Entries", "value": people_entered_with_violation},
                "ppe_count": {"label": "Valid Entries", "value": people_entered_with_full_ppe}
            }
            
            violation_frame = LatestFrame.get()
            if violation_frame is None:
                print("⚠️ Skipping violation log - frame is not ready")
            else:
                # Handle violation in existing flow
                threading.Thread(
                    target=handle_violation,
                    args=(violation_label, violation_conf, violation_frame, metrics),
                    daemon=True
                ).start()
                
                # Save violation video
                threading.Thread(
                    target=save_violation_video,
                    args=(violation_person_id, violation_label, current_time),
                    daemon=True
                ).start()
                
                print(f"🎥 Recording violation video for person {violation_person_id}: {violation_label}")

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK
    
def on_new_sample_from_sink(sink, data):
    sample = sink.emit("pull-sample")
    if sample is None:
        print("❌ Sample is None")
        return Gst.FlowReturn.ERROR

    caps = sample.get_caps()
    width = caps.get_structure(0).get_value('width')
    height = caps.get_structure(0).get_value('height')

    buf = sample.get_buffer()
    result, mapinfo = buf.map(Gst.MapFlags.READ)
    if not result:
        print("❌ Could not map buffer")
        return Gst.FlowReturn.ERROR

    try:
        expected_size = height * width * 4  # RGBA
        if mapinfo.size < expected_size:
            print(f"❌ Buffer too small: expected {expected_size}, got {mapinfo.size}")
            return Gst.FlowReturn.OK  # Don't return ERROR, just skip

        rgba = np.ndarray(
            shape=(height, width, 4),
            dtype=np.uint8,
            buffer=mapinfo.data
        )
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        LatestFrame.set(bgr)
    finally:
        buf.unmap(mapinfo)

    return Gst.FlowReturn.OK

def main():
    # Apply camera settings
    #apply_camera_settings("/dev/video0")

    Gst.init(None)
    pipeline = Gst.Pipeline()

    # Create elements
    source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
    jpegdec = Gst.ElementFactory.make("jpegdec", "jpeg-decoder")
    convert1 = Gst.ElementFactory.make("videoconvert", "convert1")
    caps_rgba = Gst.ElementFactory.make("capsfilter", "caps_rgba")
    caps_rgba.set_property("caps", Gst.Caps.from_string("video/x-raw, format=RGBA"))
    convert2 = Gst.ElementFactory.make("nvvideoconvert", "convert2")
    caps_convert2 = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
    pgie = Gst.ElementFactory.make("nvinfer", "pgie")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv")
    nvosd = Gst.ElementFactory.make("nvdsosd", "nvosd")
    sink = Gst.ElementFactory.make("nv3dsink" if is_aarch64() else "nveglglessink", "sink")

    # Check if all elements were created successfully
    for elem in [source, jpegdec, convert1, caps_rgba, convert2, caps_convert2, 
                 streammux, pgie, tracker, nvvidconv, nvosd, sink]:
        if not elem:
            print("Failed to create a GStreamer element")
            sys.exit(1)
        pipeline.add(elem)

    # Set properties
    source.set_property("device", "/dev/video0")
    caps_convert2.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
    streammux.set_property("width", FRAME_WIDTH)
    streammux.set_property("height", FRAME_HEIGHT)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 40000)
    pgie.set_property("config-file-path", "config_infer_primary_yoloV8.txt")
    
    # Configure tracker properties
    try:
        config_parser = configparser.ConfigParser()
        config_parser.read('tracker_config.txt')
        
        for key in config_parser['tracker']:
            value = config_parser.get('tracker', key)
            property_name = key.replace('-', '_')
            
            try:
                current_value = getattr(tracker.props, property_name, None)
                property_type = type(current_value)
                
                if property_name == "gpu_id" or property_type is int:
                    tracker.set_property(property_name, config_parser.getint('tracker', key))
                elif property_type is float:
                    tracker.set_property(property_name, config_parser.getfloat('tracker', key))
                elif property_type is bool:
                    tracker.set_property(property_name, config_parser.getboolean('tracker', key))
                else:
                    tracker.set_property(property_name, value)
                    
                print(f"Setting tracker {property_name} to {value}")
            except Exception as e:
                print(f"Warning: Could not set tracker property {property_name}: {e}")
                
    except Exception as e:
        print(f"Warning: Could not load tracker config file: {e}")
        print("Using default tracker settings")
        tracker.set_property("tracker-width", 640)
        tracker.set_property("tracker-height", 480)
        tracker.set_property("gpu-id", 0)
        tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
        tracker.set_property("ll-config-file", "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvSORT.yml")
    
    sink.set_property("sync", False)

    # Link pipeline
    source.link(jpegdec)
    jpegdec.link(convert1)
    convert1.link(caps_rgba)
    caps_rgba.link(convert2)
    convert2.link(caps_convert2)

    # Link NVMM output to nvstreammux sink
    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = caps_convert2.get_static_pad("src")
    if not sinkpad or not srcpad:
        print("❌ Failed to get sink or source pad for linking streammux")
        sys.exit(1)
    srcpad.link(sinkpad)

    # Link rest of the pipeline with tracker
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvidconv)
    nvvidconv.link(nvosd)
    tee = Gst.ElementFactory.make("tee", "tee")
    queue_display = Gst.ElementFactory.make("queue", "queue_display")
    queue_appsink = Gst.ElementFactory.make("queue", "queue_appsink")
    convert3 = Gst.ElementFactory.make("nvvideoconvert", "convert3")
    capsfilter = Gst.ElementFactory.make("capsfilter", "appsink_caps")
    capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw, format=RGBA"))
    appsink = Gst.ElementFactory.make("appsink", "appsink")

    # Configure appsink for OpenCV access
    appsink.set_property("emit-signals", True)
    appsink.set_property("sync", False)
    appsink.set_property("drop", True)
    appsink.set_property("max-buffers", 1)

    appsink.connect("new-sample", on_new_sample_from_sink, None)

    # Add and link elements
    for elem in [tee, queue_display, queue_appsink, convert3, capsfilter, appsink]:
        pipeline.add(elem)

    nvosd.link(tee)
    tee.link(queue_display)
    queue_display.link(sink)

    tee.link(queue_appsink)
    queue_appsink.link(convert3)
    convert3.link(capsfilter)
    capsfilter.link(appsink)

    # Setup bus and loop
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, None)

    # Add probe to OSD sink pad
    osdsinkpad = nvosd.get_static_pad("sink")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    print("🎬 Starting DeepStream PPE Detection pipeline with tracking and video recording...")
    print("📊 Tracking PPE compliance and people counting")
    print(f"🎥 Video recordings will be saved to: {VIOLATION_VIDEOS_DIR}")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop = GLib.MainLoop()
        loop.run()
    except KeyboardInterrupt:
        print("❗ Interrupted by user, stopping pipeline.")
        print("📈 Final tracking statistics:")
        print(f"  Total people entered: {people_entered}")
        print(f"  Total people exited: {people_exited}")
        print(f"  People currently inside: {people_inside}")
        print(f"  Violation entries: {people_entered_with_violation}")
        print(f"  Valid PPE entries: {people_entered_with_full_ppe}")
        
        for cls_id, class_name in CLASS_NAMES.items():
            count = len(seen_ids_by_class[cls_id])
            if count > 0:
                print(f"  {class_name}: {count} unique objects tracked")
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
