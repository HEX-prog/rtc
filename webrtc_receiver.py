"""
WebRTC Receiver Module - Ultra-Low Latency Implementation
Handles WebRTC stream reception with frame capture and statistics tracking
"""

import time
import threading
import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple
import queue


class WebRTCStreamReceiver:
    """
    WebRTC stream receiver with ultra-low latency optimizations.
    Uses aiortc and asyncio for non-blocking reception.
    """
    
    def __init__(self, stun_servers: list = None):
        """
        Initialize WebRTC receiver.
        
        Args:
            stun_servers: List of STUN server URLs (optional)
        """
        self.peer_id = None
        self.running = False
        self.connected = False
        
        # Frame buffer for minimal latency
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Statistics tracking
        self.frame_count = 0
        self.fps = 0.0
        self.frame_times = deque(maxlen=60)  # Last 60 frames for FPS calculation
        self.rtt_ms = 0.0  # Round trip time
        self.jitter_ms = 0.0  # Jitter
        self.packet_loss = 0  # Lost packets
        self.buffer_delay_ms = 0.0  # Jitter buffer delay
        
        self.last_fps_update = time.time()
        self.frames_since_last_update = 0
        
        # WebRTC specific
        self.stun_servers = stun_servers or ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]
        self.pc = None  # RTCPeerConnection
        self.call = None
        
        # Thread safe stats queue
        self.stats_queue = queue.Queue()
        
        # Try to import aiortc
        try:
            from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer
            self.RTCPeerConnection = RTCPeerConnection
            self.RTCConfiguration = RTCConfiguration
            self.RTCIceServer = RTCIceServer
            self.aiortc_available = True
        except ImportError:
            print("[WebRTC] aiortc not installed. Install with: pip install aiortc av")
            self.aiortc_available = False
        
        # Try to import peerjs alternative (aiortc is preferred)
        try:
            import asyncio
            self.asyncio = asyncio
            self.loop = None
            self.loop_thread = None
        except ImportError:
            print("[WebRTC] asyncio not available")
        
        print("[WebRTC] Receiver initialized")
    
    def generate_peer_id(self) -> str:
        """
        Generate a unique Peer ID (simulating PeerJS behavior).
        In real implementation with aiortc, we'd use proper signaling.
        
        Returns:
            str: Generated Peer ID
        """
        import uuid
        self.peer_id = str(uuid.uuid4())[:8]
        return self.peer_id
    
    def start(self):
        """Start the WebRTC receiver."""
        if not self.aiortc_available:
            print("[WebRTC] aiortc not available, using fallback mode")
            return False
        
        try:
            # Generate peer ID
            self.generate_peer_id()
            print(f"[WebRTC] Peer ID generated: {self.peer_id}")
            
            # Start async event loop in separate thread
            self.loop = self.asyncio.new_event_loop()
            self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
            self.loop_thread.start()
            
            self.running = True
            print("[WebRTC] Receiver started")
            return True
        except Exception as e:
            print(f"[WebRTC] Error starting receiver: {e}")
            return False
    
    def _run_event_loop(self):
        """Run asyncio event loop in background thread."""
        try:
            self.asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        except Exception as e:
            print(f"[WebRTC] Event loop error: {e}")
    
    def setup_peer_connection(self):
        """Setup RTCPeerConnection with ultra-low latency settings."""
        if not self.aiortc_available or not self.loop:
            return False
        
        try:
            # Create RTCConfiguration with STUN servers
            config = self.RTCConfiguration(
                iceServers=[self.RTCIceServer(urls=self.stun_servers)]
            )
            
            # Create peer connection
            self.pc = self.RTCPeerConnection(configuration=config)
            
            # Set up stream handler
            @self.pc.on("track")
            async def on_track(track):
                print(f"[WebRTC] Track received: {track.kind}")
                if track.kind == "video":
                    await self._handle_video_track(track)
            
            @self.pc.on("connectionstatechange")
            async def on_connection_state():
                print(f"[WebRTC] Connection state: {self.pc.connectionState}")
                self.connected = self.pc.connectionState == "connected"
            
            self.connected = True
            return True
        except Exception as e:
            print(f"[WebRTC] Error setting up peer connection: {e}")
            return False
    
    async def _handle_video_track(self, track):
        """Handle incoming video track with frame extraction."""
        print("[WebRTC] Video track handler started")
        
        try:
            while self.running and self.pc.connectionState == "connected":
                try:
                    # Receive frame from WebRTC
                    frame = await track.recv()
                    
                    if frame:
                        # Convert to OpenCV format
                        img = frame.to_ndarray(format="bgr24")
                        
                        # Update frame statistics
                        current_time = time.time()
                        self.frame_times.append(current_time)
                        self.frame_count += 1
                        self.frames_since_last_update += 1
                        
                        # Update FPS every 0.5 seconds
                        elapsed = current_time - self.last_fps_update
                        if elapsed >= 0.5:
                            self.fps = self.frames_since_last_update / elapsed
                            self.frames_since_last_update = 0
                            self.last_fps_update = current_time
                        
                        # Store frame thread-safely
                        with self.frame_lock:
                            self.latest_frame = img.copy()
                    
                    # Small sleep to prevent CPU spinning
                    await self.asyncio.sleep(0.001)
                
                except Exception as e:
                    print(f"[WebRTC] Frame reception error: {e}")
                    await self.asyncio.sleep(0.01)
        
        except Exception as e:
            print(f"[WebRTC] Track handler error: {e}")
        finally:
            print("[WebRTC] Video track handler stopped")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest received frame.
        
        Returns:
            numpy.ndarray or None: Latest frame or None if no frame available
        """
        if not self.running:
            return None
        
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None
    
    def update_statistics(self, stats_dict: dict):
        """
        Update WebRTC statistics from RTCStats.
        Called periodically to track network performance.
        
        Args:
            stats_dict: Dictionary with RTCStats data
        """
        try:
            # Extract RTT (Round Trip Time)
            if 'rtt' in stats_dict:
                self.rtt_ms = stats_dict['rtt'] * 1000
            
            # Extract jitter
            if 'jitter' in stats_dict:
                self.jitter_ms = stats_dict['jitter'] * 1000
            
            # Extract packet loss
            if 'packetsLost' in stats_dict:
                self.packet_loss = stats_dict['packetsLost']
            
            # Extract jitter buffer delay
            if 'jitterBufferDelay' in stats_dict:
                if stats_dict.get('jitterBufferEmittedCount', 0) > 0:
                    self.buffer_delay_ms = (
                        stats_dict['jitterBufferDelay'] / 
                        stats_dict['jitterBufferEmittedCount'] * 1000
                    )
        except Exception as e:
            print(f"[WebRTC] Error updating statistics: {e}")
    
    def get_statistics(self) -> dict:
        """
        Get current stream statistics.
        
        Returns:
            dict: Statistics dictionary with FPS, RTT, jitter, etc.
        """
        return {
            'peer_id': self.peer_id,
            'connected': self.connected,
            'fps': round(self.fps, 1),
            'frame_count': self.frame_count,
            'rtt_ms': round(self.rtt_ms, 1),
            'jitter_ms': round(self.jitter_ms, 1),
            'packet_loss': self.packet_loss,
            'buffer_delay_ms': round(self.buffer_delay_ms, 1),
            'has_frame': self.latest_frame is not None
        }
    
    def stop(self):
        """Stop the WebRTC receiver."""
        try:
            self.running = False
            self.connected = False
            
            if self.pc:
                # Close peer connection asynchronously
                if self.loop and self.loop.is_running():
                    future = self.asyncio.run_coroutine_threadsafe(self.pc.close(), self.loop)
                    try:
                        future.result(timeout=5)
                    except Exception as e:
                        print(f"[WebRTC] Error closing peer connection: {e}")
                self.pc = None
            
            if self.loop_thread and self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
                self.loop_thread.join(timeout=2)
            
            print("[WebRTC] Receiver stopped")
        except Exception as e:
            print(f"[WebRTC] Error stopping receiver: {e}")


class WebRTCCameraAdapter:
    """
    Adapter class to make WebRTC receiver compatible with existing camera interface.
    """
    
    def __init__(self, use_fallback: bool = True):
        """
        Initialize WebRTC camera adapter.
        
        Args:
            use_fallback: If True and aiortc unavailable, provide stub implementation
        """
        self.receiver = WebRTCStreamReceiver()
        self.use_fallback = use_fallback
        self.running = True
        self.last_frame = None
        
        # Try to start receiver
        if not self.receiver.start():
            if use_fallback:
                print("[WebRTC] Running in fallback mode (no actual WebRTC)")
            else:
                raise RuntimeError("WebRTC receiver failed to start and fallback disabled")
    
    def get_peer_id(self) -> str:
        """Get the Peer ID for connection."""
        return self.receiver.peer_id or "PENDING"
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get latest frame from WebRTC stream.
        
        Returns:
            numpy.ndarray or None: Latest frame
        """
        frame = self.receiver.get_latest_frame()
        
        if frame is not None:
            self.last_frame = frame
            return frame
        
        # Return last valid frame if no new frame available (for stability)
        return self.last_frame
    
    def get_statistics(self) -> dict:
        """Get stream statistics."""
        return self.receiver.get_statistics()
    
    def stop(self):
        """Stop the camera."""
        self.running = False
        self.receiver.stop()


# Standalone statistics reporter for debugging
def start_stats_reporter(receiver: WebRTCStreamReceiver, interval: float = 2.0):
    """
    Start a background thread that periodically reports WebRTC statistics.
    
    Args:
        receiver: WebRTCStreamReceiver instance
        interval: Reporting interval in seconds
    """
    def report_stats():
        while receiver.running:
            try:
                stats = receiver.get_statistics()
                print(
                    f"[WebRTC Stats] FPS: {stats['fps']}, "
                    f"RTT: {stats['rtt_ms']}ms, "
                    f"Jitter: {stats['jitter_ms']}ms, "
                    f"Buffer: {stats['buffer_delay_ms']}ms, "
                    f"Loss: {stats['packet_loss']}"
                )
                time.sleep(interval)
            except Exception as e:
                print(f"[WebRTC Stats] Error: {e}")
                break
    
    thread = threading.Thread(target=report_stats, daemon=True)
    thread.start()
    return thread
