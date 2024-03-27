import cv2
import logging
import threading
import queue
import argparse
import time

logging.basicConfig(filename='log/logfile.log', level=logging.ERROR)


class Sensor:
    def get(self):
        raise NotImplementedError("Method get() not implemented")


class SensorCam(Sensor):
    def __init__(self, camera_name, resolution):
        try:
            self.camera = cv2.VideoCapture(camera_name)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            if not self.camera.isOpened():
                raise ValueError("Camera couldn't be opened")
        except Exception as e:
            logging.error(f"Error initializing Camera: {e}")
            raise

    def get(self):
        ret, frame = self.camera.read()
        if not ret:
            logging.error("Error reading frame from camera")
        return frame

    def __del__(self):
        try:
            self.camera.release()
        except Exception as e:
            logging.error(f"Error releasing Camera: {e}")


class SensorThread:
    def __init__(self, sensor):
        self.queue = queue.Queue()
        self._sensor = sensor
        self._run = False
        self._thread = threading.Thread(target=self.run, daemon=True).start()

    def __del__(self):
        self._run = False

    def run(self):
        self._run = True
        while self._run:
            data = self._sensor.get()
            if self.queue.empty():
                self.queue.put(data)


class SensorX(Sensor):
    def __init__(self, delay):
        self._delay = delay
        self._data = 0

    def get(self):
        time.sleep(self._delay)
        self._data += 1
        return self._data


class WindowImage:

    def show(self, img):
        cv2.imshow('cam', img)
        if cv2.waitKey(1) == ord('q'):
            return True
        return False

    def __del__(self):
        cv2.destroyWindow('cam')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('camera_name', type=int, help='Name of the camera device')
    parser.add_argument('resolution', type=str, help='Resolution of the camera (e.g., 1280x720)')
    parser.add_argument('display_frequency', type=int, help='Frequency of displaying images')
    return parser.parse_args()


def main():
    args = parse_arguments()
    camera_name = args.camera_name
    resolution = tuple(map(int, args.resolution.split('x')))
    display_frequency = args.display_frequency

    sensor_x1 = SensorThread(SensorX(0.01))
    sensor_x2 = SensorThread(SensorX(0.1))
    sensor_x3 = SensorThread(SensorX(1))
    sensor_cam = SensorThread(SensorCam(camera_name, resolution))
    window_image = WindowImage()
    cam_frame = None
    sensor_x1_data = 0
    sensor_x2_data = 0
    sensor_x3_data = 0
    frame_count = 0

    try:
        while True:
            frame_count += 1
            if not sensor_cam.queue.empty():
                cam_frame = sensor_cam.queue.get()
            if not sensor_x1.queue.empty():
                sensor_x1_data = sensor_x1.queue.get()
            if not sensor_x2.queue.empty():
                sensor_x2_data = sensor_x2.queue.get()
            if not sensor_x3.queue.empty():
                sensor_x3_data = sensor_x3.queue.get()

            if cam_frame is not None:
                cv2.putText(cam_frame, f'Sensor1 data: {sensor_x1_data} Sensor2 data: {sensor_x2_data} Sensor3 data: {sensor_x3_data}',
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2,
                            cv2.LINE_AA)
                if frame_count % display_frequency == 0:
                    if window_image.show(cam_frame):
                        break
    finally:
        del sensor_cam
        del sensor_x1
        del sensor_x2
        del sensor_x3
        del window_image


if __name__ == "__main__":
    main()
