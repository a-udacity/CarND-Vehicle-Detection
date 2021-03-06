import numpy as np
from functools import partial

from moviepy.editor import VideoFileClip

from detection import slide_window, search_windows, draw_boxes, average_boxes, combine_boxes, Window
from vehicleclassifier import VehicleClassifier

process_properties = dict(
    window_parameters=[
        {'x_start_stop': [None, None], 'y_start_stop': [400, 500], 'xy_window': (96, 96), 'xy_overlap': (0.75, 0.75)},
        {'x_start_stop': [None, None], 'y_start_stop': [400, 500], 'xy_window': (144, 144), 'xy_overlap': (0.75, 0.75)},
        {'x_start_stop': [None, None], 'y_start_stop': [430, 550], 'xy_window': (192, 192), 'xy_overlap': (0.75, 0.75)},
        {'x_start_stop': [None, None], 'y_start_stop': [460, 580], 'xy_window': (192, 192), 'xy_overlap': (0.75, 0.75)},
    ],
    video_to_process='project_video.mp4',
    video_processed='project_video_processed.mp4'
)


class Process:
    def __init__(self):
        self.classifier = VehicleClassifier()
        self.classifier.load_model()
        self.window = Window()

    @staticmethod
    def _create_windows(image):
        return [
            window
            for window_parameters in process_properties['window_parameters']
            for window in slide_window(image,
                                       x_start_stop=window_parameters['x_start_stop'],
                                       y_start_stop=window_parameters['y_start_stop'],
                                       xy_window=window_parameters['xy_window'],
                                       xy_overlap=window_parameters['xy_overlap'])
            ]

    @staticmethod
    def _normalize(image):
        return image.astype(np.float32) / 255

    def _find_likely_windows(self, image):
        windows = self._create_windows(image)
        parameters = self.classifier.properties
        return search_windows(image, windows, self.classifier.model, self.classifier.x_scaler,
                              color_space=parameters['color_space'],
                              spatial_size=parameters['spatial_size'],
                              hist_bins=parameters['hist_bins'],
                              orient=parameters['orient'],
                              pix_per_cell=parameters['pix_per_cell'],
                              cell_per_block=parameters['cell_per_block'],
                              hog_channel=parameters['hog_channel'])

    def process_image(self, image, video=False):
        draw_image = np.copy(image)
        image = self._normalize(image)

        likely_windows = self._find_likely_windows(image)

        if video:
            likely_windows = combine_boxes(likely_windows, image.shape)

        results, self.window.score = average_boxes(likely_windows, self.window.score, image.shape)
        draw_image = draw_boxes(draw_image, likely_windows, color=(255, 0, 0), thick=6)
        bounding_boxes = draw_boxes(draw_image, results, color=(0, 0, 255), thick=3)

        if video:
            return bounding_boxes
        else:
            return bounding_boxes, likely_windows

    def process_video(self, video_to_process=process_properties['video_to_process'],
                      video_processed=process_properties['video_processed']):
        clip = VideoFileClip(video_to_process)
        process_video_function = partial(self.process_image, video=True)
        white_clip = clip.fl_image(process_video_function)
        white_clip.write_videofile(video_processed, audio=False)
