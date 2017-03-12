import glob
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label
from detection import make_heatmap, draw_boxes, combine_boxes, average_boxes

from process import Process


class PlotHeatMap:
    def __init__(self):
        self.process = Process()

    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # Return updated heatmap
        return heatmap  # Iterate through list of bboxes

    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return img

    def plot_measurements(self, image, image_path, save=False):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.suptitle(image_path, fontsize=50)
        f.tight_layout()
        ax1.set_title('Original Image', fontsize=25)
        ax1.imshow(image)
        window_img, box_list = self.process.process_image(image)
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)

        # Add heat to each box in box list
        heat = self.add_heat(heat, box_list)
        heat = self.apply_threshold(heat, 1)
        heatmap = np.clip(heat, 0, 255)
        labels = label(heatmap)
        heatmap = self.apply_threshold(heatmap, 2)
        labels = label(heatmap)
        ax2.imshow(labels[0], cmap='gray')
        ax2.set_title('Heat Map', fontsize=25)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        if save:
            name = os.path.basename(image_path)
            f.savefig(
                './output_images/' + '{}{}'.format(name.split('.')[0],
                                                   '_heat_measurement.jpg'))  # save the figure to file
            plt.close(f)  # close the figure

    def plot_images(self, image, image_path, save=False):
        # Read in image similar to one shown above
        # image = mpimg.imread('test_images/test1.jpg')
        window_img, box_list = self.process.process_image(image)
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)

        # Add heat to each box in box list
        heat = self.add_heat(heat, box_list)

        # Apply threshold to help remove false positives
        heat = self.apply_threshold(heat, 1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = self.draw_labeled_bboxes(np.copy(image), labels)

        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()

        if save:
            name = os.path.basename(image_path)
            fig.savefig(
                './output_images/' + '{}{}'.format(name.split('.')[0], '_heat_maps.jpg'))  # save the figure to file
            plt.close(fig)  # close the figure

    def plot_all_heatmap_images(self, save=False):
        images = glob.glob('test_images/*.jpg')
        for image_path in images:
            img = mpimg.imread(image_path)
            self.plot_images(img, image_path, save)

    def plot_all_measurements(self, save=False):
        images = glob.glob('test_images/*.jpg')
        for image_path in images:
            img = mpimg.imread(image_path)
            self.plot_measurements(img, image_path, save)


class PlotWindow:
    def __init__(self):
        self.process = Process()

    def draw_sliding_window(self, img):
        windows = self.process._find_likely_windows(img)
        return draw_boxes(img, windows)

    def plot_images(self, image, image_path, process_function, save=False, title='Sliding Window Image'):
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

    def plot_all_test_images(self, process_function, title):
        images = glob.glob('test_images/*.jpg')
        for image_path in images:
            img = mpimg.imread(image_path)
            self.plot_images(img, image_path, process_function, True, title)

    def draw_bounding_box(self, image):
        image = self.process._normalize(image)
        likely_windows = self.process._find_likely_windows(image)
        likely_windows = combine_boxes(likely_windows, image.shape)
        results, self.process.window.score = average_boxes(likely_windows, self.process.window.score, image.shape)
        draw_image = draw_boxes(image, likely_windows, color=(255, 0, 0), thick=6)
        return draw_image


from visualize import PlotHeatMap, PlotWindow


heat_map = PlotHeatMap()
### Visualize Heat Map images
heat_map.plot_all_heatmap_images(True)
### Visualize Heat Measurement images
heat_map.plot_all_measurements(True)


plot_window = PlotWindow()
### Visualize Sliding Windows
f = lambda image: plot_window.draw_sliding_window(image)
plot_window.plot_all_test_images(f, 'Sliding Window Image')

### Visualize Bounding Box
f = lambda image: plot_window.draw_bounding_box(image)
plot_window.plot_all_test_images(f, 'Bounding Box')

print("Done")
