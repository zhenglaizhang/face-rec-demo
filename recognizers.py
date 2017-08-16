from sklearn.base import BaseEstimator
from sklearn.metrics import precision_score, recall_score

import cv2


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
    def __init__(self, radius=None, neighbors=None, grid_x=None, grid_y=None, threshold=None):
        self._model = cv2.face.createLBPHFaceRecognizer(radius, neighbors, grid_x, grid_y, threshold)

    def fit(self, X, y):
        self._model.train(X, y)

    def predict(self, T):
        return [self._model.predict(T[i]) for i in range(0, T.shape[0])]
