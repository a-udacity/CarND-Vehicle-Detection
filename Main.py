import glob
import os

import cv2
import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label

from skimage.feature import hog

from vehicleclassifier import VehicleClassifier
from detection import make_heatmap, get_hog_features, _transform_to_color_space, draw_boxes, combine_boxes, \
    average_boxes
from process import Process

# clf = VehicleClassifier()
# clf.create_features()
# clf.load_features()
#
#
# clf.create_model()
#
# process = Process()
# processor_function = process.process_image
#
# process.process_video()


#### ALL Visualization code below ######

process = Process()


def draw_sliding_window(img):
    windows = process._find_likely_windows(img)
    return draw_boxes(img, windows)



def plot_images(image, image_path, process_function, save=False, title='Sliding Window Image'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.suptitle(image_path, fontsize=50)
    f.tight_layout()
    ax1.set_title('Original Image', fontsize=25)
    ax1.imshow(image)
    annotated_image = process_function(image)
    ax2.imshow(annotated_image)
    ax2.set_title(title, fontsize=25)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    if save:
        name = os.path.basename(image_path)
        f.savefig(
            './output_images/' + '{}{}'.format(name.split('.')[0], '_bounding_box.jpg'))  # save the figure to file
        plt.close(f)  # close the figure


def plot_all_test_images(process_function, title):
    images = glob.glob('test_images/*.jpg')
    for image_path in images:
        if i == 1:
            break
        img = mpimg.imread(image_path)
        plot_images(img, image_path, process_function, True, title)


# f = lambda image: draw_sliding_window(image)
#
# plot_all_test_images(f, 'Sliding Window Image')

def draw_bounding_box(image):
    image = process._normalize(image)
    likely_windows = process._find_likely_windows(image)
    likely_windows = combine_boxes(likely_windows, image.shape)
    results, process.window.score = average_boxes(likely_windows, process.window.score, image.shape)
    draw_image = draw_boxes(image, likely_windows, color=(255, 0, 0), thick=6)
    return draw_image


# f = lambda image: draw_bounding_box(image)
#
# plot_all_test_images(f, 'Bounding Box')

from visualizeheatmap import PlotHeatMap
PlotHeatMap().plot_all_heatmap_images()

print("Done")