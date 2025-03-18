import sys
import keras
from glob import glob
import tensorflow as tf
#tf.config.run_functions_eagerly(True)
import numpy as np
from data_preparation import DataPreparation
from utils import buildMagicPoint
from constants import MP_INPUT_SHAPE, MP_BATCH_SIZE, detection_confidences, eta




class CornerDetectionAveragePrecision(keras.metrics.Metric):
    def __init__(self, dtype=None, name=None):
        super().__init__(dtype, name)
        self.batchPrecisions = self.add_weight(shape=(detection_confidences.shape[0],), initializer="zeros")
        self.batchRecalls = self.add_weight(shape=(detection_confidences.shape[0],), initializer="zeros")
        self.mAP = self.add_weight(shape=(), initializer="zeros")
        self.mLE = self.add_weight(shape=(), initializer="zeros")
        
        self.instancePrecisions = self.add_weight(shape=(detection_confidences.shape[0],), initializer="zeros")
        self.instanceRecalls = self.add_weight(shape=(detection_confidences.shape[0],), initializer="zeros")
        self.localizationError = self.add_weight(shape=(), initializer="zeros")

    
    
    @tf.function(input_signature=((
        tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
        tf.TensorSpec(shape=MP_INPUT_SHAPE, dtype=tf.float32)
    ),))
    def cornerDetectionPrecision(self, instance):        
        # Removing [0, 0] coordinates from ground truth points which were added just to equalize the batch size
        groundTruthPoints = tf.boolean_mask(
            instance[0],
            tf.reduce_any(instance[0] != 0, axis=-1)
        )[tf.newaxis, ...]
        gt_points_length = tf.cast(tf.shape(groundTruthPoints)[1], tf.float32)
        index = 0
        n_detection_confidences = detection_confidences.shape[0]
        self.localizationError.assign(0.)
        for confidence in detection_confidences:
            # Getting Points where probability of pointness is greater then detection confidence
            predictedPoints = tf.cast(
                tf.sparse.from_dense(tf.greater_equal(instance[1], confidence)).indices[:, :-1], 
                tf.float32
            )[:, tf.newaxis, :]
            pp_points_length = tf.shape(predictedPoints)[0]
            # To account for special cases where FP and FN gets 0
            if tf.logical_and(
                tf.equal(gt_points_length, 0),
                tf.equal(pp_points_length, 0)
            ):
                precision = tf.constant(1, tf.float32)
                recall = tf.constant(1, tf.float32)
                n_detection_confidences -= 1
                
            elif tf.logical_or(
                tf.logical_and(
                    tf.equal(gt_points_length, 0),
                    tf.not_equal(pp_points_length, 0)
                ),
                tf.logical_and(
                    tf.not_equal(gt_points_length, 0),
                    tf.equal(pp_points_length, 0)
                )
            ):
                precision = tf.constant(0, tf.float32)
                recall = tf.constant(0, tf.float32)
                
                n_detection_confidences -= 1
            else:
                # distances between predicted and ground truth points 
                distances = tf.sqrt(
                    tf.reduce_sum(
                        tf.square(
                            tf.subtract(
                                predictedPoints,
                                groundTruthPoints
                            )
                        ),
                        axis=-1
                    )
                )
                # minimum distances for each predicted point
                minimums = tf.reduce_min(distances, axis=-1)
                # Boolean Mask for distances that are less then eta  
                correctness_mask = tf.less_equal(minimums, eta)
                # Localization error for mean distance (for this confidence) that are less then eta
                # Aggregating values at all confidences to calculate mLE outside the loop
                
                
                localizationError = tf.reduce_mean(tf.boolean_mask(minimums, correctness_mask))
                #tf.print(f"Localization Error at {confidence.numpy()}: ", localizationError)
                if tf.math.is_nan(localizationError):
                    n_detection_confidences -= 1
                else:
                    self.localizationError.assign_add(localizationError)
                # Points that are at (eta or less) far away from their respective ground truth points
                correct_points = tf.boolean_mask(
                    tf.squeeze(predictedPoints, axis=1), 
                    correctness_mask
                )
                
                TP = tf.cast(tf.shape(correct_points)[0], tf.float32)
                FP = tf.cast(tf.shape(minimums)[0], tf.float32) - TP
                FN = gt_points_length - TP
                
                precision = tf.divide(TP, tf.add(TP, FP))
                recall = tf.divide(TP, tf.add(TP, FN))

            # Writing all precision values (at all confidences) for this instance 
            self.instancePrecisions[index].assign(precision)
            self.instanceRecalls[index].assign(recall)  
            index += 1
         
        # Aggregating all precisions along their respective indices
        self.batchPrecisions.assign_add(self.instancePrecisions)
        self.batchRecalls.assign_add(self.instanceRecalls)
        
        #tf.print(f"Instance Localization Error: {self.localizationError.numpy()}")
        
        return tf.divide(self.localizationError, n_detection_confidences)
    
    
    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None, None, 2], dtype=tf.float32),
        tf.TensorSpec(shape=[None] + MP_INPUT_SHAPE, dtype=tf.float32)  
    ))
    def update_state(self, y_true, y_pred, sample_weight=None):
        localizationErrors = tf.map_fn(
            fn=self.cornerDetectionPrecision,
            elems=(y_true, y_pred),
            fn_output_signature=tf.TensorSpec((), tf.float32)
        )
        
        # Remove nans from localizationErrors
        localizationErrors = tf.boolean_mask(localizationErrors, ~tf.math.is_nan(localizationErrors))
        
        
        # Averaging precisions for each detection confidence
        self.batchPrecisions.assign(tf.divide(self.batchPrecisions, MP_BATCH_SIZE))
        self.batchRecalls.assign(tf.divide(self.batchRecalls, MP_BATCH_SIZE))
        # Averaging precisions of all detection confidences
        self.mAP.assign(tf.reduce_mean(self.batchPrecisions))
        self.mLE.assign(tf.reduce_mean(localizationErrors))
    
    
    def result(self):
        return {
            "mAP": self.mAP,
            "mLE": self.mLE,
            "precisions": self.batchPrecisions,
            "recalls": self.batchRecalls
        }
   
        
    def reset_state(self):
        self.batchPrecisions.assign(tf.zeros_like(detection_confidences))
        self.batchRecalls.assign(tf.zeros_like(detection_confidences)) 
        self.mAP.assign(0.)
        self.mLE.assign(0.)



    


if __name__ == "__main__":
    dataPreparation = DataPreparation(MP_BATCH_SIZE)
    filenames = glob("dataset/tfrecords/train/*")
    dataset = dataPreparation.loadDataset(filenames)
    iterator = dataset.__iter__()
    
    for _ in range(10):    
        images, points, bins = iterator.__next__()
        
        # Model initialization
        magicPoint = buildMagicPoint(MP_INPUT_SHAPE)
        
        # Model calling
        modelOutputs = magicPoint(images)
        
        CDAP = CornerDetectionAveragePrecision()
        CDAP.update_state(points, modelOutputs["finalOutput"])
        
        for key, value in CDAP.result().items():
            print(key, ":    ", value.numpy())
            
        CDAP.reset_state()
        
    
    
    """y_true = tf.constant([
        [
            [34., 89.],
            [56., 43.],
            [0., 123.],
            [119., 159.],
            [0., 0.],
            [0., 0.]        
        ],
        [
            [100., 100.],
            [10., 10.],
            [0., 159.],
            [119., 0.],
            [45., 76.],
            [32., 44.]
        ],
        [
            [45., 98.],
            [103., 7.],
            [1., 70.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ],
        [
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ],
        [
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ]
           
    ], tf.float16)


    modelOutput = tf.Variable(tf.random.uniform((y_true.shape[0], 120, 160, 1), minval=0, maxval=0.5))

    #modelOutput[0, 110, 159, 0].assign(0.51)
    modelOutput[0, 34, 89, 0].assign(0.51)
    modelOutput[0, 56, 43, 0].assign(0.61)
    modelOutput[0, 0, 123, 0].assign(0.71)

    modelOutput[1, 119, 0, 0].assign(0.51)
    modelOutput[1, 45, 76, 0].assign(0.61)
    modelOutput[1, 32, 44, 0].assign(0.71)
    
    modelOutput[2, 45, 98, 0].assign(0.51)
    modelOutput[2, 103, 7, 0].assign(0.61)
    modelOutput[2, 1, 70, 0].assign(0.71)
    
    modelOutput[3, 45, 98, 0].assign(0.51)
    modelOutput[3, 103, 7, 0].assign(0.61)
    modelOutput[3, 1, 70, 0].assign(0.71)
    
    
    CDAP = CornerDetectionAveragePrecision()
    CDAP.update_state(y_true, modelOutput)
    
    print(CDAP.result()['mAP'].numpy())
    CDAP.reset_state()
    print(CDAP.result()['mAP'].numpy())"""