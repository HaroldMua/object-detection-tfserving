import time
import cv2
from base_camera import BaseCamera
from tfserving_inference import Detection


class Camera(BaseCamera):
    # video_source = "rtsp://admin:admin@192.168.11.103:8554/live"
    # video_source = 0

    def __init__(self, feed_type, device, video_source):
        Camera.set_video_source(video_source)
        super(Camera, self).__init__(feed_type, device)

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def opencv_frames(device):
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            _, frame = camera.read()
            # img = cv2.flip(img, 1)
            Detection.detect_object(frame)

            cam_id = device
            yield cam_id, frame
