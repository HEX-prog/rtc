import time
import numpy as np
import mss
import cv2
import dxcam
from config import config
import threading
import queue

# ... (diğer mevcut import ve sınıflarınız)

# WebRTC desteği için
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
    import av
except ImportError:
    RTCPeerConnection = None  # aiortc opsiyonel
    print("[WebRTC] aiortc bulunamadı, WebRTC yakalayıcı kapalı.")

class WebRTCCamera:
    """
    WebRTC kamera sınıfı (PeerJS/browser tabanlı yayını almak için).
    FPS/Delay ölçümü ve Peer ID desteği içerir.
    """
    def __init__(self, region=None, target_fps=60):
        self.region = region
        self.target_fps = target_fps
        self.frame_queue = queue.Queue(maxsize=2)
        self.peer_id = self._generate_peer_id()
        self.connected = False
        self.stats = {"fps": 0, "latency_ms": 0}
        self.frame_count = 0
        self.start_time = time.time()
        self._fps_update_ts = time.time()

    def _generate_peer_id(self):
        import random, string
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))

    def get_peer_id(self):
        return self.peer_id

    def get_latest_frame(self):
        try:
            frame = self.frame_queue.get(timeout=0.05)
            return frame
        except Exception:
            return None

    def push_frame(self, frame_bgr):
        now = time.time()
        self.frame_count += 1
        elapsed = now - self.start_time
        if now - self._fps_update_ts > 1.0:
            self.stats["fps"] = int(self.frame_count / elapsed)
            self._fps_update_ts = now
        try:
            if not self.frame_queue.full():
                self.frame_queue.put(frame_bgr, block=False)
        except Exception:
            pass

    def get_metrics(self):
        return self.stats

    def is_connected(self):
        return self.connected

# --- Diğer mevcut sınıflar burada devam ediyor ---

# get_camera fonksiyonu güncellemesi

def get_camera():
    mode = config.capturer_mode.lower()
    if mode in ["webrtc", "web_rtc"]:
        cam = WebRTCCamera(target_fps=60)
        return cam, None
    # ... eski camera çeşitleriniz ...
    if mode == "mss":
        region = get_region()
        cam = MSSCamera(region)
        return cam, region
    elif mode == "ndi":
        cam = NDICamera()
        return cam, None
    elif mode == "dxgi":
        region = get_region()
        cam = DXGICamera(region)
        return cam, region
    elif mode in ["capturecard", "capture_card"]:
        region = get_region()
        cam = CaptureCardCamera(region)
        return cam, region
    elif mode == "udp":
        cam = UDPCamera(None)
        return cam, None
    else:
        raise ValueError(f"Unknown capturer_mode: {config.capturer_mode}")
