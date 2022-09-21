import os
import cv2
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "Resources") # grab image directory path

cascPath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

recognizer = cv2.face_LBPHFaceRecognizer.create()

current_id = 0
label_ids = {}

x_train = []
y_labels = []


for root, dir, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("PNG") or file.endswith("JPG"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower() # set label to folder name
            print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id # add label to dictionary
                current_id += 1 # value of dict
            id_ = label_ids[label]
            #print(label_ids)
            #y_label.append(label)
            #x.append.append(path)
            pil_image = Image.open(path).convert("L")
            size = (550, 550)
            res_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(res_image, "uint8") # convert image to numpy array for training
            #print(image_array)
            faces = faceCascade.detectMultiScale(image_array, 1.1, 4)
            for (x, y, w, h) in faces:
                face = image_array[y:y + h, x:x + w]
                x_train.append(face) # append face to model
                y_labels.append(id_)  # append label

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")

