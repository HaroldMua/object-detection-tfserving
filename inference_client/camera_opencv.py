import cv2
import queue
import threading
from base_camera import BaseCamera
from tfserving_inference import Detection
import time
from inference_client import thread_control as glv


# Refer to: https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv
# bufferless VideoCapture
class VideoCapture:

    def __init__(self, video_source, device):
        self.cap = cv2.VideoCapture(video_source)
        self.q = queue.Queue()
        self.device = device
        self._stop_event = threading.Event()
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        if not self.cap.isOpened():
            raise RuntimeError('Could not start camera.')
        # 循环获取视频源的图像帧
        while True:
            # 功能：按帧读取视频
            # 返回：ret，布尔值，读取帧是否正常，True or False；frame，每一帧的图像，三维矩阵
            ret, frame = self.cap.read()
            # device的active状态为False
            if not glv.get_device_status(self.device):
                print('_reader while over')
                break
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            # 每1ms检测按键q，检测到之后break
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            self.q.put(frame)

        # 释放摄像头
        self.cap.release()
        # 关闭图像窗口
        cv2.destroyAllWindows()

    def read(self):
        return self.q.get()


class Camera(BaseCamera):
    def __init__(self, feed_type, device, video_source_dict):
        super(Camera, self).__init__(feed_type, device, video_source_dict)
        self.unique_name = (feed_type, device)

    @staticmethod
    def opencv_frames(feed_type, device, video_source):
        # when 'video_source' is a number, it must be a int type.
        if len(video_source) < 3:
            video_source = int(video_source)

        # camera = cv2.VideoCapture(video_source)
        camera = VideoCapture(video_source, device)
        print('='*20)
        print('device {} VideoCapture active'.format(device))

        unique_name = (feed_type, device)

        # camera.stop()
        # 检测图片帧
        while True:
            print('enter loop')
            print('camera threading alive?', camera.t.isAlive())
            print('BaseCamera threading alive?', BaseCamera.threads[unique_name].isAlive())
            ac = glv.get_device_status(device)
            print('device {}: ac {}'.format(device, ac))

            # 视频关闭
            if not ac:
                print('enter ac if, False.')
                # print('camera threading:', camera.t)
                # glv.stop_thread(camera.t)
                # print('camera threading alive?', camera.t.isAlive())
                print('unique_name:', unique_name)
                # print('BaseCamera threading:', BaseCamera.threads[unique_name])
                # glv.stop_thread(BaseCamera.threads[unique_name])
                # time.sleep(10)

                # print('BaseCamera threading alive?', BaseCamera.threads[unique_name].isAlive())
                # camera.cap.release()
                print('camera opened?', camera.cap.isOpened())
                break
            # 若摄像头资源关闭，则结束循环
            if not camera.cap.isOpened():
                print('device {} 的摄像头关闭'.format(device))
                break

            frame = camera.read()
            # img = cv2.flip(img, 1)
            # 检测图片帧
            Detection.detect_object(frame)

            cam_id = device
            yield cam_id, frame
