from importlib import import_module
import time
import cv2
from flask import Flask, render_template, Response


app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index1.html')


def gen(camera_stream, feed_type, device):
    """Video streaming generator function."""
    unique_name = (feed_type, device)

    num_frames = 0
    total_time = 0
    while True:
        time_start = time.time()
        cam_id, frame = camera_stream.get_frame(unique_name)   # from here get frame
        if frame is None:
            break

        num_frames += 1
        total_time += time.time() - time_start
        fps = num_frames / total_time

        # write camera name
        cv2.putText(frame, "Camera: %s" % cam_id, (int(20), int(20 * 5e-3 * frame.shape[0])), 0, 2e-3 * frame.shape[0], (255, 255, 255), 2)

        # write camera FPS
        cv2.putText(frame, "FPS: %.2f" % fps, (int(20), int(40 * 5e-3 * frame.shape[0])), 0, 2e-3 * frame.shape[0],
                    (255, 255, 255), 2)

        frame = cv2.imencode('.jpg', frame)[1].tobytes()  # Remove this line for test camera
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed/<feed_type>/<device>')
def video_feed(feed_type, device):
    """Video streaming route. Put this in the src attribute of an img tag."""
    if feed_type == "camera_opencv":
        # video_source = "rtsp://admin:admin@192.168.11.103:8554/live"
        video_source = 0
        camera_stream = import_module('camera_opencv').Camera
        return Response(gen(camera_stream=camera_stream(feed_type, device, video_source), feed_type=feed_type, device=device),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    elif feed_type == "camera_ip":
        port = 5555
        camera_stream = import_module('camera_ip').Camera
        return Response(gen(camera_stream=camera_stream(feed_type, device, port), feed_type=feed_type, device=device),
                        mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
