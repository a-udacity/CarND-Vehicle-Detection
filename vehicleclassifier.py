from glob import glob
from time import time
from functools import partial

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from detection import extract_features
import pickle as pickle_module

classifier_properties = dict(
    color_space='YCrCb',  # RGB, HSV, LUV, HLS, YUV, YCrCb
    orient=8,
    pix_per_cell=8,
    cell_per_block=2,
    hog_channel='ALL',  # 0, 1, 2, or "ALL"
    spatial_size=(16, 16),
    hist_bins=32,
    test_size=0.2,
    vehicles='./vehicles/*',
    non_vehicles='./non-vehicles/*',
    feature_pickle_name='features.pickle',
    model_pickle_name='model.pickle'
)


class VehicleClassifier:
    def __init__(self, parameters=classifier_properties):
        self.properties = parameters
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.x_scaler = None

    def create_features(self, save=True):
        vehicles = self.glob_dataset_images_from(self.properties['vehicles'])
        non_vehicles = self.glob_dataset_images_from(self.properties['non_vehicles'])
        print('Vehicles     : {}'.format(len(vehicles)))
        print('Non-vehicles : {}'.format(len(non_vehicles)))

        orient, pix_per_cell, cell_per_block = self.properties['orient'], self.properties['pix_per_cell'], \
                                               self.properties['cell_per_block']
        extract_features_partial = partial(extract_features,
                                           color_space=self.properties['color_space'],
                                           spatial_size=self.properties['spatial_size'],
                                           hist_bins=self.properties['hist_bins'],
                                           orient=self.properties['orient'],
                                           pix_per_cell=self.properties['pix_per_cell'],
                                           cell_per_block=self.properties['cell_per_block'],
                                           hog_channel=self.properties['hog_channel'])

        vehicles_features = extract_features_partial(vehicles)
        non_vehicles_features = extract_features_partial(non_vehicles)

        x = np.vstack((vehicles_features, non_vehicles_features)).astype(np.float64)
        self.x_scaler = StandardScaler().fit(x)
        x_scaled = self.x_scaler.transform(x)

        # Define the labels vector
        y = np.hstack((np.ones(len(vehicles)), np.zeros(len(non_vehicles))))

        # Split up data into randomized training and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_scaled, y,
                                                                                test_size=self.properties['test_size'])

        print(
            'Using {} orientations {} pixels per cell {} cells per block'.format(orient, pix_per_cell, cell_per_block))
        print('Feature vector length: {}'.format(self.x_train.shape[0]))

        if save:
            print('Pickling features')
            self.pickle(
                {'x_train': self.x_train, 'x_test': self.x_test, 'y_train': self.y_train, 'y_test': self.y_test,
                 'x_scaler': self.x_scaler, 'parameters': (self.properties)},
                self.properties["feature_pickle_name"]
            )

    def load_features(self):
        features = self.unpickle(self.properties["feature_pickle_name"])
        self.x_train = features['x_train']
        self.x_test = features['x_test']
        self.y_train = features['y_train']
        self.y_test = features['y_test']
        self.x_scaler = features['x_scaler']
        self.properties = features['parameters']

    def create_model(self, model_class=LinearSVC, save=True):
        self.model = model_class()
        start = time()
        self.model.fit(self.x_train, self.y_train)
        print('Training time : {:.0f}s'.format(time() - start))
        print('Test accuracy : {:.2f}%'.format(100 * self.model.score(self.x_test, self.y_test)))

        if save:
            print('Pickling model')
            self.pickle(self.model, self.properties["model_pickle_name"])

    def load_model(self):
        if self.x_test is None:
            self.load_features()
        self.model = self.unpickle(self.properties["model_pickle_name"])

    @staticmethod
    def glob_dataset_images_from(image_path):
        all_locations = []
        cars_images = glob(image_path)
        for folder in cars_images:
            all_locations += glob('{}/*.png'.format(folder))

        return all_locations

    @staticmethod
    def pickle(object_to_pickle, destination):
        with open(destination, 'wb') as file:
            pickle_module.dump(object_to_pickle, file)

    @staticmethod
    def unpickle(pickle_file):
        with open(pickle_file, 'rb') as f:
            return pickle_module.load(f)
