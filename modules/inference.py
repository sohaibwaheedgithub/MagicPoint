import sys
import cv2
import keras
from glob import glob
from tqdm import trange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from constants import *
from utils import buildMagicPoint
from data_preparation import DataPreparation
from metrics import CornerDetectionAveragePrecision



# Instantiate the Model
magicPoint = buildMagicPoint(input_shape=MP_INPUT_SHAPE)
metric = CornerDetectionAveragePrecision()

magicPoint = keras.models.load_model(f"nd_he_normal_standardized_saved_models/magicPoint_445.keras")


instance_dir = "/mnt/c/Users/SohaibWaheed/Desktop/Personal_Projects/SuperPoint/datasets/SyntheticShapes/data/draw_checkerboard"


for i in range(1, 11):
    
    image = cv2.imread(instance_dir + f"/images/test/{i}.png", cv2.IMREAD_GRAYSCALE)

    points = np.load(instance_dir + f"/points/test/{i}.npy")[np.newaxis, ...]

    outputs = magicPoint(np.expand_dims(image, axis=[0, -1]))

    predictedPoints = tf.sparse.from_dense(tf.greater_equal(outputs["finalOutput"][0], 0.30)).indices[:, :-1]

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    for point in points[0]:
        cv2.circle(
            image,
            (int(point[1]), int(point[0])),
            1,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    for point in predictedPoints:
        cv2.circle(
            image,
            (int(point[1].numpy()), int(point[0].numpy())),
            1,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )
    plt.imshow(image)
    plt.show()
