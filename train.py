import pickle
import os
from tqdm import tqdm
from shutil import copyfile
import pathlib
import logging
import json
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from skimage.transform import resize


from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def remove_files(dirs):
    files = glob.glob(dirs+'/*')
    for f in files:
        os.remove(f)

class Train():

    def __init__(self):
        self.Model = MobileNet(input_shape=(224, 224, 3), include_top=False, pooling='avg')
        self.n_clusters = 7
        self.cluster_model_file_name = "kmean-7.pkl"
        self.no_of_neighbours = 50
        #self.encoder = CNN()


    def create_required_folders(self):

        if os.path.isdir('Models'):
            pass
        else:
            os.mkdir('Models')

        if os.path.isdir('Models/embeddings'):
            pass
        else:
            os.mkdir('Models/embeddings')

        if os.path.isdir('clustered_data'):
            pass
        else:
            os.mkdir('clustered_data')

        if os.path.isdir('Models/embeddings_index'):
            pass
        else:
            os.mkdir('Models/embeddings_index')

    def kmean_clustering(self, X):
        Kmean = KMeans(n_clusters=self.n_clusters)
        return Kmean.fit(X)

    def knn(self, X):
        return NearestNeighbors(n_neighbors=self.no_of_neighbours, algorithm='ball_tree').fit(X)


    def train(self, input_data_dir, save_images_in_cluster_dir=True):

        assert len(os.listdir(input_data_dir)) >= 7, "insufficient data"

        self.create_required_folders()
        parms_file_log = {}                         # params_file_log will be used to keep track of parameters and paths of inputs and output that will used while inferencing
        parms_file_log['training_data_path'] = input_data_dir
        parms_file_log['n_clusters'] = self.n_clusters
        parms_file_log['no_of_neighbours'] = self.no_of_neighbours

        X, index_dict = self.prepare_data(input_data_dir) # output will be embedding of images

        # train the clustering model
        Kmean = self.kmean_clustering(X)
        cluster_pkl_filename = os.path.join("Models",self.cluster_model_file_name)
        with open(cluster_pkl_filename, 'wb') as file:                   # save the cluster model for inferencing
            pickle.dump(Kmean, file)

        parms_file_log['cluster_model'] = cluster_pkl_filename
        logging.info('Training Clustering Model Done')

        clu = []
        for _ in range(self.n_clusters):
            clu.append([])

        for i, label in enumerate(Kmean.labels_):
            clu[label].append(index_dict[i])

        for i in range(self.n_clusters):
            op = 'no of images in cluster {} --- {}'.format(i, len(clu[i]))
            logging.info(op)

        if save_images_in_cluster_dir:                 # saving images in respective cluster
            for i in range(self.n_clusters):
                cluster_dir = os.path.join('clustered_data', 'cluster_'+str(i))
                if not os.path.isdir(cluster_dir):
                    os.mkdir(cluster_dir)
                remove_files(cluster_dir)
                for path in clu[i]:
                    src = path
                    dst = os.path.join(cluster_dir,src.split('/')[-1])
                    copyfile(src, dst)
            #print('Images are saved in respectve clusters')
            logging.info('Images are saved in respectve clusters in clustered_data folder')

        # train a nearest neighbour model for each cluster and save the model
        for i in range(self.n_clusters):
            X_C, cluster_index = self.partiton_cluster_data(X, index_dict, clu[i])
            knn_model = self.knn(X_C)
            knn_model_name_dir = os.path.join('Models', 'cluster_tree_'+str(i)+'.pkl')

            with open(knn_model_name_dir, 'wb') as file:
                pickle.dump(knn_model, file)
                parms_file_log['knn_model_'+str(i)] = knn_model_name_dir

            index_name = os.path.join('Models/embeddings_index', 'cluster_tree_'+str(i)+'.pkl')
            with open(index_name, 'wb') as file:
                pickle.dump(cluster_index, file)
                parms_file_log['cluster_index_file_'+str(i)] = index_name

        with open('Models/parms.json', 'w') as f:
            json.dump(parms_file_log, f)

    def partiton_cluster_data(self, X, index_dict, clu):
        '''
        funcition will separate out the data for nearest neighbour algo and preserves the indexes
        '''
        reversed_index_dict = {value: key for key, value in index_dict.items()}
        X_C = []
        cluster_index = {}
        for i, f_name in enumerate(clu):
            X_C.append(X[reversed_index_dict[f_name]])
            cluster_index[i] = f_name
        X_C = np.array(X_C)
        return X_C, cluster_index

    def preprocess_image(self, image_array_nd, target_size=(224, 224)):
        '''
        will preprocess in require format for mobilenet
        '''
        image_pil = Image.fromarray(image_array_nd)
        image_pil = image_pil.resize(target_size, Image.ANTIALIAS)
        image_array = np.array(image_pil).astype('uint8')
        image_pp = preprocess_input(image_array)
        image_pp = np.array(image_pp)[np.newaxis, :]
        return image_pp

    def encode_image(self, image_array):
        '''
        will produce a embeddng for image
        '''
        image_pp = self.preprocess_image(image_array_nd = image_array)
        return self.Model.predict(image_pp)

    def make_dirs(self, img_paths_dir):
        '''
        helping function of makin directories
        '''
        img_paths = [os.path.join(img_paths_dir, path) for path in os.listdir(img_paths_dir)]
        return img_paths

    def prepare_data(self, input_data_dir, save_embeddings=True):
        X = []
        index_dict = {}
        logging.info('creation of embedding started')
        for i in tqdm(range(len(os.listdir(input_data_dir)))):
            path = os.path.join(input_data_dir, str(i) + '.jpg')
            # print("-----------", path)
            image = cv2.imread(path)
            # print(image.shape)
            #embd = self.encode_image(image_array=image)[0]
            embd = self.encode_image(image_array=image)[0]
            #embd = normalize(embd[:, np.newaxis], axis=0).ravel()
            X.append(embd)
            index_dict[i] = path

        X = np.array(X)
        #print(index_dict)
        #print('creation of embeddings done')
        logging.info('creation of embeddings done')

        if save_embeddings:
            with open('Models/embeddings/data.npy', 'wb') as f:
                np.save(f, X)

            filename = 'Models/embeddings_index/index_dict.pkl'
            #os.makedirs(os.path.dirname(filename), exist_ok=True)
            # filename = 'index_dict.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(index_dict, f)

            logging.info('embeddings and index files are saved in Model directory ')
        return X, index_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_to_training_dir', help='give path to input dir')
    args = parser.parse_args()

    k = Train()
    # input_data_dir = 'simple_data'
    #input_data_dir = 'dataset'
    k.train(args.path_to_training_dir)
