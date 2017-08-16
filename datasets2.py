#!/usr/bin/env python3

import sys
import os
import config
import logging
import numpy as np
from recognizers import *
import sklearn.model_selection as ms


class DataSetReader(object):
    def __init__(self, path, size=None):
        self._path = path
        self._size = size

    def read(self):
        c = 0
        X, y, names = [], [], {}
        for dirname, dirnames, filenames in os.walk(self._path):
            for subdirname in dirnames:
                subject_path = os.path.join(dirname, subdirname)
                for filename in os.listdir(subject_path):
                    try:
                        im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                        if self._size is not None:
                            im = cv2.resize(im, size)
                        X.append(np.asarray(im, dtype=np.uint8))
                        y.append(c)
                    except IOError as err:
                        logging.error(err)
                    except:
                        logging.error('Unexpected error:{}'.format(sys.exc_info()[0]))
                        raise
                names[c] = subdirname
                c += 1
        X, y = [np.array(l) for l in [X, y]]
        return [X, y]


if __name__ == '__main__':
    X_train, y_train = DataSetReader(config.att_faces_path).read()
    # cv = ms.StratifiedKFold(n_splits=10, random_state=42)
    estimator = FisherFaceRecognizer()
    precision_scores = ms.cross_val_score(estimator,
                                          X_train,
                                          y_train,
                                          cv=3,
                                          scoring='accuracy')
    print(precision_scores)
