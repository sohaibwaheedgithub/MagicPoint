import gc
import sys
import cv2
import math
import keras
import numpy as np
from tqdm import tqdm
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from constants import MP_INPUT_SHAPE, MP_BATCH_SIZE




class DataPreparation():
    
    def __init__(self, batchSize):
        self.batchSize = batchSize    
        self.features = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "points": tf.io.FixedLenFeature([], tf.string),
            "bins": tf.io.FixedLenFeature([], tf.string),
        }
    
    
    def parseExamples(self, example):
        parsedExample = tf.io.parse_single_example(example, self.features)
        image = tf.io.decode_png(parsedExample["image"])
        image.set_shape(MP_INPUT_SHAPE)
        points = tf.io.parse_tensor(parsedExample["points"], out_type=tf.float32)
        points.set_shape([None, 2])
        bins = tf.io.parse_tensor(parsedExample["bins"], out_type=tf.int32)
        bins.set_shape([MP_INPUT_SHAPE[0]//8, MP_INPUT_SHAPE[1]//8])
        
        return image, points, bins
        
        
        
    def loadDataset(self, filenames):
        return tf.data.TFRecordDataset(
            filenames, 
            num_parallel_reads=tf.data.AUTOTUNE
            
        ).prefetch(
            buffer_size=tf.data.AUTOTUNE   
        ).map(
            self.parseExamples,
            num_parallel_calls=tf.data.AUTOTUNE
        ).cache().padded_batch(
            self.batchSize,
            drop_remainder=True
        )
 
    
            
    def drawGrid(self, image):
        y = MP_INPUT_SHAPE[1] // 8
        x = MP_INPUT_SHAPE[0] // 8
        for i in range(8):
            cv2.line(
                image,
                (y, 0),
                (y, 120),
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )
            
            cv2.line(
                image,
                (0, x),
                (160, x),
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )
            
            y += MP_INPUT_SHAPE[1] // 8
            x += MP_INPUT_SHAPE[0] // 8
            
    
            
    def plotPoints(self, image, points, generatedPoints):
        for point in points:
            cv2.circle(
                image,
                (point[1], point[0]),
                5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
        for generatedPoint in generatedPoints:
            cv2.drawMarker(
                image,
                (int(generatedPoint[1]), int(generatedPoint[0])),
                #(255, 255, 0),
                (255, 0, 0),
                markerType=0,
                markerSize=3,
                thickness=2,
                line_type=cv2.LINE_AA
            )
        
            
        
             
    def visualizeDataset(self, dataset, model=None):
        noInstances = 25
        rows, columns = 5, 5
        figure = plt.figure(figsize=(19, 10))
        for idx, (ori_image, points, bins) in dataset.unbatch().skip(noInstances*1).take(noInstances).enumerate(start=1):  
            figure.add_subplot(rows, columns, idx.numpy())
            image = cv2.cvtColor((ori_image.numpy()[:, :, 0]).astype("uint8"), cv2.COLOR_GRAY2RGB)
            points = points.numpy().astype('int32')
            points = points[np.any(points != 0, axis=-1)]
            if not model:
                categoricalBins = keras.utils.to_categorical(bins, num_classes=65)[..., :-1]
                reformedImage = tf.nn.depth_to_space(categoricalBins[tf.newaxis, ...], block_size=8)[0]
                generatedPoints = tf.sparse.from_dense(reformedImage).indices[:, :-1]
                self.plotPoints(image, points, generatedPoints)
            else:
                outputs = model(ori_image[tf.newaxis, ...])
                predictedPoints = tf.sparse.from_dense(tf.greater_equal(outputs["finalOutput"][0], 0.30)).indices[:, :-1]
                self.plotPoints(image, [], predictedPoints)
            
            plt.imshow(image, cmap="gray")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

        
        
        
        
        
if __name__ == "__main__":
    dataPreparation = DataPreparation(batchSize=32)
    filenames = glob("dataset/tfrecords/valid/*")
     
    '''norm_layer = keras.layers.Normalization(axis=-1)
    for filename in tqdm(filenames):    
        dataset = dataPreparation.loadDataset([filename])  
        imageDataset = dataset.map(lambda images, *args: images)
        
        norm_layer.adapt(imageDataset) 
            
        del dataset, imageDataset
        gc.collect()
    
    print(norm_layer.mean.numpy())             # 126.36381
    print((norm_layer.variance).numpy())       # 3035.8782'''
    
    model_file = "nd_he_normal_standardized_saved_models/magicPoint_445.keras"
    model = keras.models.load_model(model_file)
    
    dataset = dataPreparation.loadDataset(filenames)
    dataPreparation.visualizeDataset(dataset, model)