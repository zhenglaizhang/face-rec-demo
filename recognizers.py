from sklearn.base import BaseEstimator
from sklearn.metrics import precision_score, recall_score

import cv2


# threshold: the confidence score, which allows us to set thresholds in real-life applications to limit the amount of false reads.

# Eigenfaces and Fisherfaces will produce values (roughly) in the range 0 to 20,000, with # any score below 4-5,000 being quite a confident recognition.
# LBPH works similarly; however, the reference value for a good recognition is below 50, and any value above 80 is considered as a low confidence score
# A normal custom approach would be to hold-off drawing a rectangle around a recognized face until we have a number of frames with a satisfying arbitrary confidence score

class EigenFaceRecognizer(BaseEstimator):
    def __init__(self):
        self._model = cv2.face.createEigenFaceRecognizer()

    def fit(self, X, y):
        self._model.train(X, y)

    def predict(self, T):
        return [self._model.predict(T[i]) for i in range(0, T.shape[0])]


class FisherFaceRecognizer(BaseEstimator):
    def __init__(self, num_components=None, threshold=None):
        self._model = cv2.face.createFisherFaceRecognizer()

    def fit(self, X, y):
        self._model.train(X, y)

    def predict(self, T):
        return [self._model.predict(T[i]) for i in range(0, T.shape[0])]


class LBPHFaceRecognizer(BaseEstimator):
    # todo: how to elegantly pass the default value??
    def __init__(self, radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=123):
        self._model = cv2.face.createLBPHFaceRecognizer(radius, neighbors, grid_x, grid_y, threshold)

    def fit(self, X, y):
        self._model.train(X, y)

    def predict(self, T):
        return [self._model.predict(T[i]) for i in range(0, T.shape[0])]
