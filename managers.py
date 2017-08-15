import cv2
import time
import numpy as np


class CaptureManager(object):
    def __init__(self, camera, preview_window_manager=None, mirror=False):
        self.preview_window_manager = preview_window_manager
        self.mirror = mirror
        self._capture = camera
        self._channel = 0
        self._entered_frame = False
        self._frame = None
        self._img_file_name = None
        self._video_file_name = None
        self._video_encoding = None
        self._video_writer = None

        self._start_time = None
        self._frames_elapsed = 0
        self._fps_estimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._entered_frame and self._frame is None:
            _, self._frame = self._capture.retrieve()
        return self._frame

    @property
    def is_writing_img(self):
        return self._img_file_name is not None

    @property
    def is_writing_video(self):
        return self._video_file_name is not None

    def enter_frame(self):
        """capture the next frame, if any"""
        assert not self._entered_frame, 'previous enter_frame() had no matching exit_frame()'
        if self._capture is not None:
            self._entered_frame = self._capture.grab()

    def exit_frame(self):
        """preprocessing, draw to window, write to files and release frame"""
        if self.frame is None:
            self._entered_frame = False
            return

        # update fps estimate
        if self._frames_elapsed == 0:
            self._start_time = time.time()
        else:
            time_elapsed = time.time() - self._start_time
            self._fps_estimate = self._fps_estimate / time_elapsed

        self._frames_elapsed += 1

        # draw to window if any
        if self.preview_window_manager is not None:
            if self.mirror:
                mirrored_frame = np.fliplr(self._frame).copy()
                self.preview_window_manager.show(mirrored_frame)
            else:
                self.preview_window_manager.show(self._frame)

        # write to file if any
        if self.is_writing_img:
            cv2.imwrite(self._img_file_name, self._frame)
            self._img_file_name = None

        self._write_video_frame()

        self._frame = None
        self._entered_frame = False

    def write_image(self, img_file_name):
        self._img_file_name = img_file_name

    def start_writing_video(self, filename, encoding=cv2.VideoWriter_fourcc('I', '4', '2', '0')):
        self._video_file_name = filename
        self._video_encoding = encoding

    def stop_writing_video(self):
        self._video_file_name = None
        self._video_encoding = None
        self._video_writer = None

    def _write_video_frame(self):
        if not self.is_writing_video:
            return

        if self._video_writer is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:  # capture's FPS unknown
                if self._frames_elapsed < 20:
                    # wait until more frames elapses to make estimate more stable
                    return
                else:
                    fps = self._fps_estimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._video_writer = cv2.VideoWriter(self._video_file_name, self._video_encoding, fps, size)
        self._video_writer.write(self._frame)


class Cv2WindowManager(object):
    def __init__(self, window_name, key_press_callback=None):
        self._key_press_callback = key_press_callback
        self._window_name = window_name
        self._is_window_created = False

    @property
    def is_window_created(self):
        return self._is_window_created

    def create_window(self):
        cv2.namedWindow(self._window_name)
        self._is_window_created = True

    def show(self, frame):
        cv2.imshow(self._window_name, frame)

    def destory_window(self):
        cv2.destoryWindow(self._window_name)
        self._is_window_created = False

    def process_events(self):
        key_code = cv2.waitKey(1)
        if self._key_press_callback is not None and key_code != -1:
            key_code &= 0xFF    # discard non-ASCII codes
            self._key_press_callback(key_code)
