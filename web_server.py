import os
import cv2
import numpy as np
import json
import threading
import asyncio
import yaml
import csv
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.sdp import candidate_from_sdp
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

#import det_count as det# we explicitly start its thread below
import det_vid as det
#import detx as det
THIS_DIR       = os.path.dirname(__file__)
CONFIG_PATH    = os.path.join(THIS_DIR, "config.json")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

CSV_LOG = config["log_behavior"]["csv_path"]


ROOT = os.path.dirname(__file__)
relay = MediaRelay()


class CameraTrack(VideoStreamTrack):
    """
    Pulls 640×480 BGR frames from detection.LatestFrame.buffer (no resizing needed).
    Throttles to 15 FPS to keep encoding smooth.
    """
    def __init__(self):
        super().__init__()
        self._last_ts = None

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        now = asyncio.get_event_loop().time()

        # Throttle to ~15 FPS
        if self._last_ts is not None:
            elapsed = now - self._last_ts
            if elapsed < (1/15):
                await asyncio.sleep((1/15) - elapsed)

        frame = det.LatestFrame.get()
        if frame is None:
            print("Frame is none in recv")
            black = np.zeros((675, 1080, 3), dtype="uint8")
            rgb = cv2.cvtColor(black, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        new_frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        new_frame.pts = pts
        new_frame.time_base = time_base

        self._last_ts = asyncio.get_event_loop().time()
        return new_frame


pcs = set()


async def index(request):
    path = os.path.join(ROOT, "webrtc.html")
    content = open(path, "r").read()
    return web.Response(content_type="text/html", text=content)


async def offer(request):
    params = await request.json()
    offer_sdp  = params["sdp"]
    offer_type = params["type"]

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_state_change():
        print("Connection state →", pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)

    # Add our CameraTrack (reads 640×480 annotated frames directly)
    camera = CameraTrack()
    pc.addTrack(relay.subscribe(camera))

    # 1) Apply incoming SDP offer
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=offer_sdp, type=offer_type)
    )

    # ✅ Ensure LatestFrame is initialized before continuing
    waited = 0
    while det.LatestFrame.get() is None and waited < 5:
        print(f"🕒 Waiting for DeepStream frame... ({waited:.1f}s)")
        await asyncio.sleep(0.2)
        waited += 0.2

    if det.LatestFrame.get() is None:
        print("❌ DeepStream frame not ready — aborting WebRTC offer")
        await pc.close()
        pcs.discard(pc)
        return web.Response(status=503, text="Camera feed not ready")

    # 2) Create & send back SDP answer (let VP8/VP9 be negotiated)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )


async def webrtc_offer(request):
    params = await request.json()
    offer_sdp  = params["sdp"]
    offer_type = params["type"]

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_state_change():
        print("Connection state →", pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)

    # Add track
    camera = CameraTrack()
    pc.addTrack(relay.subscribe(camera))

    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type=offer_type))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })


pcs = set()

async def webrtc_ice(request):
    try:
        data = await request.json()
        raw = data.get("candidate")

        # 1) basic validation
        if not isinstance(raw, dict):
            return web.Response(status=400, text="Invalid ICE payload")
        for key in ("candidate", "sdpMid", "sdpMLineIndex"):
            if key not in raw:
                return web.Response(status=400, text=f"Missing ICE field: {key}")

        # 2) find the active PeerConnection
        pc: RTCPeerConnection = next(iter(pcs), None)
        if pc is None:
            return web.Response(status=404, text="No active PeerConnection")

        # 3) parse with aiortc’s SDP helper
        ice = candidate_from_sdp(raw["candidate"])
        ice.sdpMid        = raw["sdpMid"]
        ice.sdpMLineIndex = raw["sdpMLineIndex"]

        print("✅ adding ICE candidate:", ice)
        await pc.addIceCandidate(ice)    # accepts full RTCIceCandidate object
        return web.Response(status=200)

    except Exception as e:
        print("❌ ICE add failed:", repr(e))
        return web.Response(status=500, text=str(e))

async def stop_stream(request):
    for pc in list(pcs):
        await pc.close()
        pcs.discard(pc)
    return web.Response(status=200)

# ─── LOG ROUTE ──────────────────────────────────────────────────────────────────
async def log(request):
    """
    Return the contents of the CSV log as JSON array.
    """
    try:
        with open(CSV_LOG, newline='') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        return web.json_response(data)
    except Exception as e:
        print("❌ /log handler error:", e)
        return web.Response(status=500, text=str(e))


async def send_violation_alert(request):
    """
    Send violation data to backend with image
    """
    try:
        data = await request.json()
        
        # Get current frame for violation image
        frame = det.LatestFrame.get()
        image_buffer = None
        
        if frame is not None:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            image_buffer = base64.b64encode(buffer).decode('utf-8')
        
        # Send to backend
        violation_data = {
            'deviceId': data.get('deviceId', '507f1f77bcf86cd799439011'),  # Default device ID
            'violationType': data.get('violationType', 'no-helmet'),
            'confidence': data.get('confidence', 85),
            'boundingBox': data.get('boundingBox'),
            'timestamp': data.get('timestamp'),
            'location': data.get('location', 'Camera 1'),
            'imageBuffer': image_buffer
        }
        
        # Here we would send to our backend API
        # For now, just log it
        print(f"📨 Sending violation alert: {violation_data['violationType']} ({violation_data['confidence']}%)")
        
        return web.json_response({'success': True})
        
    except Exception as e:
        print("❌ Violation alert error:", e)
        return web.Response(status=500, text=str(e))


def main():
    # ── STEP A: Start detection loop in the same process ─────────────────────────
    threading.Thread(target=det.main, daemon=True).start()
    print("⟳ Detection thread started. LatestFrame.buffer → 640×480 frames.")

    # ── STEP B: Launch WebRTC server ─────────────────────────────────────────────
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_post("/stream/offer", webrtc_offer)
    app.router.add_post("/stream/ice-candidate", webrtc_ice)
    #app.router.add_post("/stream/stop", stop_stream)
    app.router.add_get("/log", log)
    app.router.add_post("/violation", send_violation_alert)

    host = "0.0.0.0"
    port = 8000
    print(f"WebRTC server listening on http://{host}:{port}")
    web.run_app(app, host=host, port=port)


if __name__ == "__main__":
    main()
