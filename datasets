#!/usr/bin/env python3

# ./004_face_rec_fisher_train_test.py camera

import cv2
import sys
import os
import config


def init_cascade():
    return cv2.CascadeClassifier(config.haarcascade_frontalface_default_model_path)


size = 4
fn_dir = config.att_faces_path

try:
    fn_name = sys.argv[1]
except IndexError:
    print('please provide a name')
    sys.exit(1)

path = os.path.join(fn_dir, fn_name)
if not os.path.isdir(path):
    os.mkdir(path)
im_width, im_height = 112, 92
print(im_width)
print(im_height)
cascade = init_cascade()

pin = sorted([int(n[:n.find('.')]) for n in os.listdir(path)
              if n[0] != '.'] + [0])[-1] + 1
print(pin)

print("\n\033[94mThe program will save 20 samples. \
Move your head around to increase while it runs.\033[0m\n")
count = 0
pause = 0
count_max = 20

cam = cv2.VideoCapture(0)
while count < count_max:
    rval = False
    while not rval:
        rval, frame = cam.read()
        if not rval:
            print('Failed to read from webcam, trying again')

    height, width, channels = frame.shape

    # vertical flip
    frame = cv2.flip(frame, 1, 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # scale down to speed up
    small = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

    # detect faces
    faces = cascade.detectMultiScale(small, 1.1, 5)

    # only consider largest face
    faces = sorted(faces, key=lambda x: x[3])
    if faces:
        face_i = faces[0]
        x, y, w, h = [v * size for v in face_i]
        face = gray[y:y + h, x: x + w]
        face_resize = cv2.resize(face, (im_width, im_height))

        # draw and label
        cv2.rectangle(frame, (x, y), (x + w, y + w), (0, 255, 0), 3)
        cv2.putText(frame, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        # remove FPs
        if w * 6 < width or h * 6 < height:
            print('Ignoring small faces...')
        else:
            # To create diversity, only save every fith detected image
            if pause == 0:
                print('Saving training sample...', count + 1, '/', count_max)
                cv2.imwrite('%s/%s.png' % (path, pin), face_resize)
                pin += 1
                count += 1
                pause = 1

    if pause > 0:
        pause = (pause + 1) % 5
    cv2.imshow('Sampling...', frame)
    key = cv2.waitKey(10) & 0xFF
    if key == 27:
        break
