import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd

class BerryCluster:
    """Class to cluster berries in bounding boxes"""
    def __init__(self, berry_dir='datasets/wgisd_annotations', 
                        data_dir='datasets/wgisd_annotations'):
        self.berry_dir = berry_dir
        self.data_dir = data_dir
        self.berries_coord = None
        self.bounding_boxes = None
        self.combined_array = None

    def load_gt_masks(self, npz_file):
        data = np.load(f'{self.data_dir}/{npz_file}')
        self.combined_array = np.zeros(data['arr_0'].shape[0:2])
        bunch_count = data['arr_0'].shape[2]
        for i in range(bunch_count):
            self.combined_array += data['arr_0'][:,:,i]

    def load_berry_coordinates(self, txt_file):
        self.berries_coord = pd.read_csv(f'{self.berry_dir}/{txt_file}', sep='\t', header=None)
        self.berries_coord.columns = ['x', 'y']

    def load_bounding_boxes(self, txt_file):
        self.bounding_boxes = pd.read_csv(f'{self.data_dir}/{txt_file}', sep=' ', header=None)
        self.bounding_boxes.columns = ['class', 'center_x', 'center_y', 'w', 'h']
        img_height, img_width = self.combined_array.shape
        self.bounding_boxes['x'] = (self.bounding_boxes['center_x'] - self.bounding_boxes['w']/2) * img_width
        self.bounding_boxes['y'] = (self.bounding_boxes['center_y'] - self.bounding_boxes['h']/2) * img_height
        self.bounding_boxes['w'] *= img_width
        self.bounding_boxes['h'] *= img_height
        self.bounding_boxes.index.name = 'id'

    def associate_berries_to_boxes(self):
        """Associate each berry to its bounding box"""
        self.berries_coord['in_bb'] = -1
        for i, berry in self.berries_coord.iterrows():
            for j, box in self.bounding_boxes.iterrows():
                if (berry['x'] >= box['x']) and (berry['x'] <= box['x'] + box['w']) and (berry['y'] >= box['y']) and (berry['y'] <= box['y'] + box['h']):
                    self.berries_coord.at[i, 'in_bb'] = j
                    break

    def perform_kmeans(self, n_clusters=3):
        """Perform k-means clustering on the berries in each bounding box"""
        results = []
        for bb_id in self.bounding_boxes.index:
            points_in_bb = self.berries_coord[self.berries_coord['in_bb'] == bb_id][['x', 'y']]
            if len(points_in_bb) >= n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(points_in_bb)
                iter_coord = kmeans.cluster_centers_
            elif len(points_in_bb) > 0:
                kmeans = KMeans(n_clusters=len(points_in_bb), random_state=0, n_init='auto').fit(points_in_bb)
                iter_coord = kmeans.cluster_centers_
            else: # coloca um ponto no centro da bounding box, sem aplicar k-means
                center_x = self.bounding_boxes.at[bb_id, 'center_x'] * self.combined_array.shape[1]
                center_y = self.bounding_boxes.at[bb_id, 'center_y'] * self.combined_array.shape[0]
                iter_coord = np.array([[center_x, center_y]])
                print(iter_coord)

            for i, center in enumerate(iter_coord):
                # Adicionar os centros ao novo DataFrame
                results.append({
                    'bb_id': bb_id,
                    'center_x': center[0],
                    'center_y': center[1]
                })
        self.df_cluster = pd.DataFrame(results)
        return self.df_cluster

if __name__ == '__main__':
    for file in os.listdir('datasets/wgisd_annotations'):
        if file.endswith('.npz') and 'SYH_2017-04-27_1304' in file:
            print(file)
            berry_cluster = BerryCluster()
            berry_cluster.load_gt_masks(file)
            berry_cluster.load_berry_coordinates(file.replace('.npz', '-berries.txt'))
            berry_cluster.load_bounding_boxes(file.replace('.npz', '.txt'))
            berry_cluster.associate_berries_to_boxes()
            three_points = berry_cluster.perform_kmeans(n_clusters=3)
            print(three_points)
            three_points.to_excel(file.replace('.npz', '-3points_.xlsx'), index=False)
            single_point = berry_cluster.perform_kmeans(n_clusters=1)
            single_point.to_excel(file.replace('.npz', '-1point_.xlsx'), index=False)
