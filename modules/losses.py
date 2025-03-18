import sys
import keras
from glob import glob
import tensorflow as tf
from utils import buildMagicPoint
from data_preparation import DataPreparation
from constants import MP_BATCH_SIZE, MP_INPUT_SHAPE


    
if __name__ == "__main__":
    
    dataPreparation = DataPreparation(MP_BATCH_SIZE)
    filenames = glob("dataset/tfrecords/train/*")
    dataset = dataPreparation.loadDataset(filenames)
    magicPoint = buildMagicPoint(MP_INPUT_SHAPE)
    iterator = dataset.__iter__()
    images, points, bins = iterator.__next__()
    
    magicPointOutput = magicPoint(images)
    
    loss = keras.losses.sparse_categorical_crossentropy(
        bins, 
        magicPointOutput["interestPointDecoderOutput"]
    )
    