#!/usr/bin/env python3

# ./003_face_rec_fisher.py

import csv
import cv2
import numpy as np
import os
import sys
import prepare_train_test as pcsv
import config

size = 4


def init_cascade():
    return cv2.CascadeClassifier(config.haar_defaul_model_path)


def split_train_test(data_dir, train_path, test_path, max=5):
    train, test = pcsv.split_train_test(data_dir, max)
    pcsv.write_dicts_to_csv(train, train_path)
    pcsv.write_dicts_to_csv(test, test_path)


def prepare_data(csv_file):
    imgs, ls, ns = [], [], {}
    with open(csv_file) as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            imgs.append(cv2.imread(row['image'], cv2.IMREAD_GRAYSCALE))
            label = int(row['label'])
            ls.append(label)
            ns[label] = row['name']

    return imgs, ls, ns


def train_fisher(images, labels, model_threshold=1100):
    images, labels = [np.array(l) for l in [images, labels]]
    model = cv2.face.createFisherFaceRecognizer(threshold=model_threshold)
    # model = cv2.face.createEigenFaceRecognizer(threshold=model_threshold)
    model.train(images, labels)
    return model


def train_eigenface(images, labels, model_threshold=1100):
    images, labels = [np.array(l) for l in [images, labels]]
    model = cv2.face.createEigenFaceRecognizer(threshold=model_threshold)
    model.train(images, labels)
    return model


def train_lbph(images, labels, model_threshold=1100):
    images, labels = [np.array(l) for l in [images, labels]]
    model = cv2.face.createLBPHFaceRecognizer(threshold=model_threshold)
    # model = cv2.face.createEigenFaceRecognizer(threshold=model_threshold)
    model.train(images, labels)
    return model


def rec_through_camera(cascade, model, im_width=112, im_height=92):
    cam = cv2.VideoCapture(0)
    while True:
        rval = False
        while not rval:
            rval, frame = cam.read()
            if not rval:
                print('Failed to read from webcam, trying again')

        # vertical flip (optional)
        frame = cv2.flip(frame, 1, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # scale down to speed up
        small = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

        # detect faces
        faces = cascade.detectMultiScale(small, 1.1, 5)

        for i, face_i in enumerate(faces):
            # pre-process
            x, y, w, h = [v * size for v in face_i]
            face = gray[y: y + h, x: x + w]
            face_resize = cv2.resize(face, (im_width, im_height))

            # predict
            prediction = model.predict(face_resize)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            if prediction[0] == -1:
                predicted_label = 'UNKNOWN'
            else:
                predicted_label = names[prediction[0]]

            cv2.putText(frame, '%s - %.0f' % (predicted_label, prediction[1]), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        cv2.imshow('Predicting...', frame)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break


def rec_thru_images(cascade, model, test_images, test_labels, test_names, im_width=112, im_height=92,
                    model_name='default'):
    tp, fp, miss = 0, 0, 0
    for ti, test_image in enumerate(test_images):
        faces = cascade.detectMultiScale(test_image, 1.1, 5)
        for i, face_i in enumerate(faces):
            x, y, w, h = [v for v in face_i]
            # face = test_image[y: y + h, x: x + w]
            face = test_image
            face_resize = cv2.resize(face, (im_height, im_width))
            prediction = model.predict(face_resize)
            nbr_actual = test_labels[ti]
            if prediction[0] == -1:
                # not in lib
                # print("{} - {}/{} is not Recognized".format(model_name, test_names[nbr_actual], test_labels[ti]))
                prediction_label = 'UNKNOWN'
                miss += 1
            else:
                # recognized, test if it's correct
                if nbr_actual == prediction[0]:
                    prediction_label = test_names[prediction[0]]
                    tp += 1
                    # print("{} - {}/{} is Correctly Recognized with distance {}".format(model_name,
                    #                                                                    prediction_label,
                    #                                                                    nbr_actual,
                    #                                                                    prediction[1]))
                else:
                    prediction_label = 'FALSE'
                    fp += 1
                    # print("{} - {}/{} is Incorrectly Recognized as {}/{}".format(model_name,
                    #                                                              test_names[nbr_actual],
                    #                                                              nbr_actual,
                    #                                                              test_names[prediction[0]],
                    #                                                              prediction[0]))
            cv2.putText(face, '%s - %.0f' % (prediction_label, prediction[1]),
                        (x - 5, y - 5),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 255, 0))

            cv2.imshow('Recognizing Faces', cv2.resize(face, (im_height * size, im_width * size)))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    return tp, fp, miss


if __name__ == '__main__':
    try:
        target = sys.argv[1]
    except IndexError:
        print('Please specify target parameter: [image | camera]'
              'between 6 & 9)')
        sys.exit(1)

    cascade = init_cascade()
    if target == 'camera':
        print('Preprocessing data')
        train_path = os.path.join(config.temp_path, 'att_faces_train.csv')
        test_path = os.path.join(config.temp_path, 'att_faces_test.csv')

        # for camera stream, all existing images should be used as train set
        split_train_test(config.att_faces_path, train_path, test_path, max=1000)
        images, labels, names = prepare_data(train_path)

        print('Training...')
        model = train_fisher(images, labels)

        print('Predicting thru camera stream (make sure you have put your label inside att_faces dir')
        rec_through_camera(cascade, model)
    if target == 'image':
        train_path = os.path.join(config.temp_path, 'att_faces_train.csv')
        test_path = os.path.join(config.temp_path, 'att_faces_test.csv')
        for train_size in range(5, 10):
            split_train_test(config.att_faces_path, train_path, test_path, max=train_size)
            train_images, train_labels, train_names = prepare_data(train_path)
            test_images, test_labels, test_names = prepare_data(test_path)
            models = {
                'eigenface': train_fisher(train_images, train_labels, model_threshold=1100),
                'fisher': train_fisher(train_images, train_labels, model_threshold=1100),
                'lbph': train_lbph(train_images, train_labels, model_threshold=95)
            }

            for k, v in models.items():
                tp, fp, miss = rec_thru_images(cascade, v, test_images, test_labels, test_names, model_name=k)
                # ignore the face which are not detected...
                print('{},{},{},{}'.format(k, train_size, tp / (fp + tp), tp / (miss + tp)))
    print('DONE')
