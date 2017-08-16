import cv2
import config


class HaarCascadeDetector(object):
    def __init__(self, model_path):
        self._cascade_model = cv2.CascadeClassifier(model_path)

    def detect(self, img, resize, neighbors):
        rois = self._cascade_model.detectMultiScale(img, resize, neighbors)
        return rois

    def resize_detect(self, img, scale, resize, neighbors):
        rsz = cv2.resize(img, scale)
        self.detect(rsz, resize, neighbors)

    def detect_largest(self, img, resize, neighbors):
        rois = self.detect(img, resize, neighbors)
        if rois:
            rois = sorted(rois, key=lambda f: f[3])
            return rois[0]
        else:
            return None


class FrontalFaceDetector(HaarCascadeDetector):
    def __init__(self):
        super().__init__(config.haarcascade_frontalface_default_model_path)


class EyeDetector(HaarCascadeDetector):
    def __init__(self):
        super().__init__(config.haarcascade_eye_model_path)
