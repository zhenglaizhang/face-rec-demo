import cv2
import config


class FaceDetector(object):
    pass


class HaarFaceDetector(FaceDetector):
    def __init__(self):
        self._cascade = cv2.CascadeClassifier(config.haarcascade_frontalface_default_model_path)

    def detect(self, img, resize, neighbors):
        faces = self._cascade.detectMultiScale(img, resize, neighbors)
        return faces

    def resize_detect(self, img, scale, resize, neighbors):
        rsz = cv2.resize(img, scale)
        self.detect(rsz, resize, neighbors)

    def detect_largest(self, img, resize, neighbors):
        faces = self.detect(img, resize, neighbors)
        if faces:
            faces = sorted(faces, key=lambda f: f[3])
            return faces[0]
        else:
            return None
