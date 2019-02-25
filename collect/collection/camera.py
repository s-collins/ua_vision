import cv2 as cv

CAMERA_DEVICE_ID = 0
# OpenCV class VideoCapture's constructor requires integer ID of device.


class Camera:
    """Abstraction for the web cam."""

    def __init__(self):
        self.capture = cv.VideoCapture(CAMERA_DEVICE_ID)
        self.open = self.capture.isOpened()

    def __del__(self):
        self.capture.release()

    def good(self):
        """Returns true iff camera is open and ready."""
        return self.open

    def get_frame(self):
        """Returns an OpenCV matrix from the webcam.

        If the camera is not open, this method raises a RuntimeError.
        """
        if self.open:
            ret, frame = self.capture.read()
            return frame
        else:
            return RuntimeError('Camera could not get frame.')


class FakeCamera:

    def __init__(self):
        pass

    def good(self):
        return True

    def get_frame(self):
        return cv.imread('image.jpg', cv.IMREAD_COLOR)
