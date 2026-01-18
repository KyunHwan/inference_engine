import threading
import cv2
import time

class RBRSCamera:
    class singleCamera:
        def __init__(self, device_id, brightness=None, contrast=None, saturation=None, exposure=None):
            self.id = device_id
            self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise ValueError(f"Camera with device ID {device_id} could not be opened.")
 
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            if brightness is not None:
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            if contrast is not None:
                self.cap.set(cv2.CAP_PROP_CONTRAST, contrast)
            if saturation is not None:
                self.cap.set(cv2.CAP_PROP_SATURATION, saturation)
            if exposure is not None:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
 
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Camera {device_id} initialized: {w}x{h}@{fps}fps FOURCC={fourcc_str}")
            
        def get_frame(self):
            ret, frame = self.cap.read()
            if not ret:
                return None
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame = cv2.resize(frame, (800, 600))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
 
        def release(self):
            if self.cap is not None:
                self.cap.release()
    
    def __init__(self, device_id1=0, device_id2=2, brightness=None, contrast=None, saturation=None, exposure=None):
        self.camera1 = self.singleCamera(device_id1, brightness, contrast, saturation, exposure)
        self.camera2 = self.singleCamera(device_id2, brightness, contrast, saturation, exposure)
        self.lock = threading.Lock()
        self.frame1 = None
        self.frame2 = None
        self.running = True
        self.thread = threading.Thread(target=self.update_frames, daemon=True)
        self.thread.start()
        self.merged_frame = None
        self.device_id1 = device_id1
        self.device_id2 = device_id2
 
    def update_frames(self):
        while self.running:
            f1 = self.camera1.get_frame()
            f2 = self.camera2.get_frame()

            if f1 is None:
                print(f"Warning: device {self.device_id1} frames is None.")
            if f2 is None:
                print(f"Warning: device {self.device_id2} frames is None.")

            with self.lock:
                old_frame1 = self.frame1
                old_frame2 = self.frame2
                
                if f1 is not None:
                    self.frame1 = f1
                else:
                    self.frame1 = None
                if f2 is not None:
                    self.frame2 = f2
                else:
                    self.frame2 = None
                
                if self.frame1 is not None and self.frame2 is not None:
                    self.merged_frame = cv2.hconcat([self.frame2, self.frame1])
                else:
                    self.merged_frame = None
                
                del old_frame1, old_frame2
            
    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.update_frames, daemon=True)
            self.thread.start()
            
    def get_image(self):
        with self.lock:
            if self.merged_frame is not None:
                return self.merged_frame.copy()
            return None
    
    def visualize(self):
        while self.running:
            with self.lock:
                f1 = self.frame1
                f2 = self.frame2
 
            frame_to_show = None
 
            if f1 is not None and f2 is not None:
                frame_to_show = self.merged_frame
            elif f1 is not None:
                frame_to_show = f1
            elif f2 is not None:
                frame_to_show = f2
 
            if frame_to_show is not None:
                cv2.imshow('Merged Cameras', frame_to_show)
 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break
 
    def stop(self):
        if not self.running:
            return
        self.running = False
        self.thread.join()
        self.camera1.release()
        self.camera2.release()