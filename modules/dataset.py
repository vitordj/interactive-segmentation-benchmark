import json
import numpy as np
import zlib
import base64
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd


class ImageAnnotations:
    """Class that consolidates all the information of an image annotation, regardless of the number or type of objects.
    :param image_name: Not the relative path, only image name with extension. (.JPG, .jpg, .png, etc.)"""

    def __init__(self, image_name: str):
        self.image_name = image_name
        self.box_annotations = []
        self.point_annotations = []
        self.gt_image = None
        self.image_pred = {} # Dicionário para armazenar as predições

    def load_image(self, image_path: str):
        """Opens an image from a given path.
        :param image_path: Path to the image.
        """
        self.image = cv2.imread(image_path)

    def read_supervisely(self, json_path: str):
        """Reads a json file from Supervisely and stores the information in the ImageAnnotation object.
        :param json_path: Path to the json file.
        """
        with open(json_path) as json_file:
            data = json.load(json_file)
        annotations = data # ['annotation']
        self.image_size = annotations['size']
        self.segmentation_annotations = []
        self.bb_id = 0  # Inicializando o ID da bounding box
        self.point_id = 0  # Inicializando o ID do ponto

        for obj in annotations['objects']:
            if obj['geometryType'] == 'point':
                point = {}
                point['class'] = obj['classTitle']
                point['coord'] = obj['points']['exterior'][0]
                point['is_positive'] = False if 'negativo' in obj['classTitle'] else True # check if point is negative (adapt if you need)
                point['is_aux'] = True if 'aux' in obj['classTitle'] else False # check if point is auxiliary, interesting for comparison between different point sets
                point['id'] = self.point_id
                self.point_id += 1
                self.point_annotations.append(point)
            elif obj['geometryType'] == 'rectangle':
                box = {}
                box['class'] = obj['classTitle']   
                box['coords'] = obj['points']['exterior'] # [[x1, y1], [x2, y2]]
                box['id'] = self.bb_id
                box['points_inside'] = []
                self.bb_id += 1
                self.box_annotations.append(box)
            elif obj['geometryType'] == 'bitmap':
                # origin and data
                bitmap = {}
                bitmap['class'] = obj['classTitle']
                bitmap['origin'] = obj['bitmap']['origin']
                bitmap['bitmap_gt'] = obj['bitmap']['data']
                bitmap['np_gt'] = self.base64_2_mask(bitmap['bitmap_gt'])
                self.segmentation_annotations.append(bitmap)
            else:
                print('Unknown geometry type: {}'.format(obj['geometryType']))
        self._associate_points_supervisely()
        self._calculate_relative_point_positions()
        self._reconstruct_supervisely_gt()

    def read_wgisd(self, npz_path: str, excel_sufx = '-1point'):
        """Reads a npz file from WGisd and stores the information in the ImageAnnotation object.
        :param npz_path: Path to the npz file.
        :param sufx: Suffix of the txt file that contains the excel file with calculated k-means points."""
        
        data = np.load(npz_path)
        self.image_size = data['arr_0'].shape[0:2]
        # Combine all masks layers into a single layer
        self.gt_image = self._wgisd_load_gt(npz_path)
        self._wgisd_load_BBs(npz_path.replace('.npz', '.txt'))
        self._wgisd_load_points(npz_path.replace('.npz', excel_sufx + '.xlsx'))
        # Create a list of points coordinates relative to each bounding box
        self._calculate_relative_point_positions()
        print('WGisd annotations loaded.')


    def _wgisd_load_BBs(self, txt_file):
        bbs = pd.read_csv(f'{txt_file}', sep=' ', header=None)
        bbs.columns = ['class', 'center_x', 'center_y', 'w', 'h']
        img_height, img_width = self.image_size
        # convert to [[x1, y1], [x2, y2]]
        bbs['x1'] = ((bbs['center_x'] - bbs['w']/2) * img_width).round(0).astype(int)
        bbs['y1'] = ((bbs['center_y'] - bbs['h']/2) * img_height).round(0).astype(int)
        bbs['x2'] = ((bbs['center_x'] + bbs['w']/2) * img_width).round(0).astype(int)
        bbs['y2'] = ((bbs['center_y'] + bbs['h']/2) * img_height).round(0).astype(int)
        bbs.index.name = 'id'
        for id in bbs.index:
            box = {}
            box['coords'] = [[bbs.loc[id, 'x1'], bbs.loc[id, 'y1']],
                            [bbs.loc[id, 'x2'], bbs.loc[id, 'y2']]]
            box['id'] = id
            self.box_annotations.append(box)
            self.box_annotations[id]['points_inside'] = []
    
    def _wgisd_load_points(self, excel_file):
        """Load points created previouly using K-mean clustering and 
        associate them with its corresponding BB.
        
        In this case (wgisd) all points are positive and non-auxiliary."""
        points = pd.read_excel(excel_file)
        # self.box_annotations[bb_id]['points_inside'] = []
        for id in points.index:
            point = {}
            point['coord'] = points.loc[id, 'center_x'], points.loc[id, 'center_y']
            point['is_positive'] = True
            point['is_aux'] = False
            point['id'] = id
            bb_id = points.loc[id, 'bb_id']
            point['bounding_box'] = bb_id
            self.point_annotations.append(point)
            # Associate point to BB 
            self.box_annotations[bb_id]['points_inside'].append(id)

    @staticmethod
    def _wgisd_load_gt(npz_file):
        data = np.load(npz_file)
        gt_image = np.zeros(data['arr_0'].shape[0:2])
        bunch_count = data['arr_0'].shape[2]
        for i in range(bunch_count):
            gt_image += data['arr_0'][:,:,i]
        return gt_image

    def bb_image_2_np(self, bb):
        """Converts a bounding box annotation to a numpy array."""
        x1, y1 = bb['coords'][0]
        x2, y2 = bb['coords'][1]
        return self.image[y1:y2, x1:x2]

    @staticmethod
    def base64_2_mask(s):
        """Converts Supervisely's bitmap annotation to a numpy array."""
        z = zlib.decompress(base64.b64decode(s))
        n = np.frombuffer(z, np.uint8)
        mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
        return mask

    @staticmethod
    def _is_point_inside_bbox(point, bbox):
        """Checks if a given point is inside a bounding box."""
        x, y = point
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def _associate_points_supervisely(self):
        """Associates each point to a bounding box and updates the point dictionary with a 'bounding_box' key."""
        for point in self.point_annotations:
            for box in self.box_annotations:
                # Verificar se o ponto está dentro da bounding box
                if self._is_point_inside_bbox(point['coord'], box['coords']):
                    if 'bounding_box' in point.keys():
                        print('Point already has a bounding box associated.')
                        break
                    point['bounding_box'] = box['id']  # Atribuir o ID da bounding box ao ponto
                    box['points_inside'].append(point['id'])
                    break  # Uma vez que o ponto é associado a uma bounding box, sair do loop interno

    def _calculate_relative_point_positions(self):
        """Calculates the relative position of the point inside its associated bounding box."""
        for point in self.point_annotations:
            if 'bounding_box' in point:
                bbox = next(box for box in self.box_annotations if box['id'] == point['bounding_box'])
                point['coord_bb'] = [point['coord'][0] - bbox['coords'][0][0], 
                                    point['coord'][1] - bbox['coords'][0][1]]

    def _reconstruct_supervisely_gt(self):
        """Reconstructs the ground truth segmentation image from bitmap annotations."""
        # Suponha que o tamanho da imagem original seja (height, width)
        height, width = self.image_size['height'], self.image_size['width']
        
        # Crie uma matriz vazia para a imagem de segmentação
        self.gt_image = np.zeros((height, width), dtype=np.uint8)

        for annotation in self.segmentation_annotations:
            mask = annotation['np_gt']
            origin = annotation['origin']
            
            # Coloque a máscara na posição apropriada da matriz
            h, w = mask.shape
            self.gt_image[origin[1]:origin[1]+h, origin[0]:origin[0]+w] = mask * 255

    def reconstruct_prediction_mask(self, predictions_list:list[np.array], prediction_name:str):
        """Reconstructs the prediction mask from a list of predictions. Will help us compare different predictions easily.
        :param predictions_list: List of predictions, each prediction is a numpy array. List order must be the same as the bounding boxes IDs.
        :param prediction_name: Name (key) of the prediction."""
        height, width, _ = self.image.shape
        self.image_pred[prediction_name] = np.zeros((height, width))
        for i, box in enumerate(self.box_annotations):
            prediction = predictions_list[i]
            x1, y1 = box['coords'][0]
            x2, y2 = box['coords'][1]
            # verifica se a predição tem o mesmo tamanho que a bounding box, se não, redimensiona
            if prediction.shape[0] != y2-y1 or prediction.shape[1] != x2-x1:
                prediction = cv2.resize(prediction, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
            self.image_pred[prediction_name][y1:y2, x1:x2] = prediction

    def validate_supervisely_annotations(self):
        """Valida as anotações garantindo que cada bounding box contenha um ponto não auxiliar e dois auxiliares."""
        for box in self.box_annotations:
            non_aux_points = 0
            aux_points = 0

            for point_id in box['points_inside']:
                point = next(p for p in self.point_annotations if p['id'] == point_id)
                if point['is_aux']:
                    aux_points += 1
                else:
                    non_aux_points += 1

            if non_aux_points != 1 or aux_points != 2:
                print(f"Bounding box with ID {box['id']} (Class: {box['class']}) has {non_aux_points} non-auxiliary points and {aux_points} auxiliary points!")


def main():
    anot = ImageAnnotations('nikon_fila_10_auto_DSC_0109.JPG')
    anot.read_supervisely(json_path='experiment\nikon_fila_10_auto_DSC_0109.JPG.json')
    print('Segmentation annotation example:')
    print(anot.segmentation_annotations[0])
    print('Boounding box annotation example:')
    print(anot.box_annotations[:2])
    anot.validate_supervisely_annotations()

if __name__ == '__main__':
    main()