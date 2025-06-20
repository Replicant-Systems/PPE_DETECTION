import os
import cv2
import uuid
import json
import boto3
from datetime import datetime
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# === AWS IoT MQTT CONFIG ===
AWS_IOT_ENDPOINT = ""
DEVICE_ID = ""
PROJECT_ID = ""
S3_BUCKET = ""

# === MQTT TLS FILES ===
CA_PATH = "AmazonRootCA1.pem"
CERT_PATH = "cert.pem.crt"
KEY_PATH = "private.pem.key"

AWS_ACCESS_KEY_ID = "AKIAU44WZPSAYSJND4EJ"
AWS_SECRET_ACCESS_KEY = "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"
AWS_REGION = "ap-south-1"

# === Init AWS S3 ===
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# === Init AWS IoT MQTT Client ===
mqtt_client = AWSIoTMQTTClient(DEVICE_ID)
mqtt_client.configureEndpoint(AWS_IOT_ENDPOINT, 8883)
mqtt_client.configureCredentials(CA_PATH, KEY_PATH, CERT_PATH)

# MQTT Client Configuration (retries, timeout, queueing)
mqtt_client.configureOfflinePublishQueueing(-1)  # Infinite offline Publish queueing
mqtt_client.configureDrainingFrequency(2)  # Draining: 2 Hz
mqtt_client.configureConnectDisconnectTimeout(10)  # 10 sec
mqtt_client.configureMQTTOperationTimeout(5)  # 5 sec

mqtt_client.connect()
print("✅ Connected to AWS IoT via MQTT")

def handle_violation(class_name: str, confidence: float, frame, metrics: dict):
    timestamp = datetime.utcnow().isoformat()
    uid = str(uuid.uuid4())
    filename = f"{uid}-{timestamp}.jpg"
    local_path = f"/tmp/{filename}"
    s3_key = f"{DEVICE_ID}/violations/{filename}"

    # 1. Save frame locally
    cv2.imwrite(local_path, frame)

    # 2. Upload to S3
    try:
        s3.upload_file(local_path, S3_BUCKET, s3_key)
        print(f"✅ Uploaded to S3: {s3_key}")
    except Exception as e:
        print(f"❌ S3 upload failed: {e}")
        return

    # 3. Publish MQTT message to AWS IoT
    payload = {
        "projectId": PROJECT_ID,
        "deviceId": DEVICE_ID,
        "timestamp": timestamp,
        "class": class_name,
        "confidence": round(confidence * 100, 1),
        "s3Key": s3_key,
        "domainMetrics": metrics
    }

    topic = f"device/{DEVICE_ID}/violations"
    mqtt_client.publish(topic, json.dumps(payload), 1)
    print(f"📡 MQTT published to topic: {topic}")

    # 4. Optional: Clean up local file
    os.remove(local_path)

