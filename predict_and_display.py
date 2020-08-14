import pickle
import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

class Predict():
    def __init__(self):

        self.Model = MobileNet(input_shape=(224, 224, 3), include_top=False, pooling='avg')

        with open('Models/parms.json', 'r') as f: # load the parms used in training which contains cluster size, indexes, data_paths etc.
            self.parms_data = json.load(f)

        with open(self.parms_data["cluster_model"], 'rb') as f:
            self.cluster_model = pickle.load(f)

        self.training_imgs_dir = self.parms_data['training_data_path']

        self.knn_trees = self.load_knn_trees()
        self.knn_index_dicts = self.load_knn_index_dicts()

    def load_knn_trees(self):
        knn_trees = []
        for i in range(self.parms_data["n_clusters"]):
            with open(self.parms_data["knn_model_" + str(i)], 'rb') as f:
                tree_model = pickle.load(f)
                knn_trees.append(tree_model)
        return knn_trees

    def load_knn_index_dicts(self):
        knn_index_dicts = []
        for i in range(self.parms_data["n_clusters"]):
            with open(self.parms_data["cluster_index_file_" + str(i)], 'rb') as f:
                index_dict = pickle.load(f)
                knn_index_dicts.append(index_dict)
        return knn_index_dicts

    def plot_images(self, no_of_images_to_display, query_image, cluster_predicted, indices, path_to_input_image):

        N = no_of_images_to_display
        cols = 3
        rows = int(np.ceil((N + 3) / 3))
        axes = []
        fig = plt.figure()

        index_dict = self.knn_index_dicts[cluster_predicted]

        imgs_names = [index_dict[idx] for idx in indices[0][:N]]
        imgs_dirs_to_display = [os.path.join(self.training_imgs_dir, pth) for pth in imgs_names]

        axes.append(fig.add_subplot(rows, cols, 1))
        query_image = query_image[:, :, ::-1]
        subplot_title = ('query image')
        axes[-1].set_title(subplot_title)
        axes[-1].axis('off')
        plt.imshow(query_image)

        for i, img_dir in enumerate(imgs_names):
            image = cv2.imread(img_dir)[:, :, ::-1]
            axes.append(fig.add_subplot(rows, cols, i + 4))
            axes[-1].axis('off')
            # subplot_title=('query image')
            # axes[-1].set_title(subplot_title)
            plt.imshow(image)

        if not os.path.isdir('output'):
            os.mkdir('output')

        plt.axis('off')
        plt.savefig(os.path.join('output',path_to_input_image.split('/')[-1]), dpi=300, bbox_inches='tight')
        plt.show()

    def display_similar_images(self, path_to_input_image, no_of_images_to_display):
        image = cv2.imread(path_to_input_image)
        # product embedding
        embdding = self.encode_image(image_array=image)[0]
        embdding = np.expand_dims(embdding, axis=0)
        # predict which cluster it belong
        cluster_predicted = self.cluster_model.predict(embdding)[0]
        # find indices of the images near the embedding
        distances, indices = self.knn_trees[cluster_predicted].kneighbors(embdding)
        self.plot_images(no_of_images_to_display, image, cluster_predicted, indices, path_to_input_image)

    def preprocess_image(self, image_array_nd, target_size=(224, 224)):
        image_pil = Image.fromarray(image_array_nd)
        image_pil = image_pil.resize(target_size, Image.ANTIALIAS)
        image_array = np.array(image_pil).astype('uint8')
        image_pp = preprocess_input(image_array)
        image_pp = np.array(image_pp)[np.newaxis, :]
        return image_pp

    def encode_image(self, image_array):
        image_pp = self.preprocess_image(image_array_nd = image_array)
        return self.Model.predict(image_pp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_to_query_image', help='give path to query_image')
    parser.add_argument('-n', '--no_of_images_to_display', help='give path to query_image')
    args = parser.parse_args()

    k = Predict()
    #path_to_input_image = '/home/uma/Desktop/projects/image_clustering/final/dataset/1678.jpg'
    #no_of_images_to_display = 10
    k.display_similar_images(args.path_to_query_image, int(args.no_of_images_to_display))