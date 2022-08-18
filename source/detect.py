import os
import cv2
import numpy as np
import logging

# import torch
from insightface.app import FaceAnalysis
from numpy.linalg import norm


class InsightFace:
    def __init__(self, name_model='buffalo_l', ctx_id=0, det_size=(640, 640),
                 threshold=0.35, path_face="face_embedding_no_user.pt", path_name="list_name_no_user.pt"):
        self.name_model = "buffalo_l"
        self.threshold = threshold
        self.path_name = path_name
        self.path_face = path_face

        self.app = FaceAnalysis(name=name_model)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        
        self.data_face_embedding = {}
        
    def compare_two_face(self, image1, image2):
        status = ''
        new_embedding1, bbox1, count_face1 = self.face_embedding(image1)
        if count_face1 != 1:
            status = 'Error! Have 2 face in image 1'
            return status, 0
        new_embedding2, bbox2, count_face2 = self.face_embedding(image2)
        if count_face2 != 1:
            status = 'Error! Have 2 face in image 2'
            return status, 0
        distance = np.dot(new_embedding1, new_embedding2.T) / (np.linalg.norm(new_embedding1) * np.linalg.norm(new_embedding2))
        status = 'Success'
        return status, distance
        
    def verify_face(self, image):
        new_embedding, bbox, count_face = self.face_embedding(image)
        if count_face == 1:
            rs_distance = dict()
            for name, emb in self.data_face_embedding.items():
                distance = np.dot(new_embedding, emb.T) / (np.linalg.norm(new_embedding) * np.linalg.norm(emb))
                rs_distance[f'{name}'] = distance

            name = max(rs_distance, key=lambda k: rs_distance.get(k))
            if rs_distance[name] > self.threshold:
                self._draw_on(image=image, bbox=bbox, label=name)
                return image, name, rs_distance[name]
            else:
                self._draw_on(image=image, bbox=bbox, label=-1)
                return image, name, -1
                
    def face_embedding(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings_image = self.app.get(img=image)
        count_face = len(encodings_image)
        if count_face == 1:
            embedding = encodings_image[0].embedding
            bbox = encodings_image[0].bbox
            return embedding, bbox, count_face
        elif count_face > 1:
            logging.warning('Have more than one face')
            return -1, -1, count_face
        else:
            logging.warning('No face')
            return -1, -1, count_face
    
    def save_data_face(self, path_images):
        for i in os.listdir(path_images):
            img = cv2.imread(os.path.join(path_images, i))
            name = i
            embedding, bbox, count_face = self.face_embedding(img)
            if count_face == 1:
                self.data_face_embedding[f'{name}'] = embedding
                
            else:
                logging.error(f'Image must be 1 face, check {i}')
    
    def _draw_on(self, image, bbox, label):
        if bbox is not None:
            bbox = bbox.astype(np.int)
            if label != -1 :
                name = label
                image = cv2.putText(
                    img=image,
                    text=name,
                    org=(bbox[0]-1, bbox[1]-4),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    thickness=2
                )
                color = (255, 0, 0)
            else:           
                image = cv2.putText(
                    img=image,
                    text="UnKnown",
                    org=(bbox[0]-1, bbox[1]-4),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    thickness=2
                )
                color = (0, 0, 255)
            image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
    