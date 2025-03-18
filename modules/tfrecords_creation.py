import sys
import keras
import numpy as np
from glob import glob
import pandas as pd
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from constants import MP_INPUT_SHAPE



def loadDirs(setName):
    dirs = np.stack(
        [
        glob(f"dataset/data/*/images/{setName}/*"), 
        glob(f"dataset/data/*/points/{setName}/*")
        ], 
        axis=1
    )
    np.random.shuffle(dirs)
    return dirs.T



testImgsDir, testPtsDir = loadDirs("test")
trainImgsDir, trainPtsDir = loadDirs("training")
validImgsDir, validPtsDir = loadDirs("validation")



dirs = {
    'train': (trainImgsDir, trainPtsDir, 100),
    'test': (testImgsDir, testPtsDir, 10),
    'valid': (validImgsDir, validPtsDir, 10)
}




def generateBins(points):
    print("points: ", points)
    x = range(0, MP_INPUT_SHAPE[0])
    y = range(0, MP_INPUT_SHAPE[1])
    X, Y = tf.meshgrid(x, y, indexing="ij")
    X, Y = X[..., tf.newaxis], Y[..., tf.newaxis]
    gridsRegion = tf.reshape(tf.cast(tf.concat([X, Y], axis=-1), tf.float32), [-1, 1, 2])
    print("gridsRegion: ", gridsRegion)
    
    binsBooleanMask = tf.reduce_all(tf.equal(gridsRegion, points[tf.newaxis, ...]), axis=-1)[..., tf.newaxis]
    print("binsBooleanMask: ", binsBooleanMask)
    sys.exit(0)
    
    binsBooleanMask = tf.broadcast_to(binsBooleanMask, [binsBooleanMask.shape[0], points.shape[0], 2]) 
    binsBooleanMask = tf.reshape(binsBooleanMask, [MP_INPUT_SHAPE[0], MP_INPUT_SHAPE[1], points.shape[0], 2])
    
    
    binsBinaryMask = tf.reduce_sum(
        tf.cast(
            tf.reduce_all(
                binsBooleanMask, 
                axis=-1
            ), 
            tf.float32
        ), 
        axis=-1
    )[tf.newaxis, ..., tf.newaxis]
    
    
    bins = tf.image.extract_patches(
        images=binsBinaryMask,
        sizes=[1, MP_INPUT_SHAPE[0] // 8, MP_INPUT_SHAPE[1] // 8, 1],
        strides=[1, MP_INPUT_SHAPE[0] // 8, MP_INPUT_SHAPE[1] // 8, 1],
        rates=[1, 1, 1, 1],
        padding="SAME"
    )
    bins = tf.reshape(bins, [-1, MP_INPUT_SHAPE[0] // 8, MP_INPUT_SHAPE[1] // 8])
    bins = tf.transpose(bins, [1, 2, 0])
    bins = tf.concat([bins, tf.ones([bins.shape[0], bins.shape[1], 1], dtype=bins.dtype)*0.5], axis=-1)
    bins = tf.argmax(bins, axis=-1, output_type=tf.uint16)
    return bins, binsBinaryMask
    
    




# To calculate bin for each point
def calculateBins(points):
    tensor = tf.math.floor(
        tf.divide(
            points, 
            tf.cast(
                tf.divide(MP_INPUT_SHAPE[:2], 8), 
                tf.float32
            )
        )
    )
    return tf.cast((tensor[:, 0] * 8) + tensor[:, 1], tf.uint16)





def preprocessPoints(points):
    points = tf.math.round(points)
    mask = tf.logical_or(
        tf.equal(points, MP_INPUT_SHAPE[0]),
        tf.equal(points, MP_INPUT_SHAPE[1])
    )
    points = tf.where(mask, points-1, points)
    points = np.unique(points, axis=0)
    np.random.shuffle(points)
    # Calculating bins row and column indices to prevent two points being ended up in the same pixel cell
    binsRowColumn = tf.map_fn(
        lambda point: tf.stack([point[0] % (MP_INPUT_SHAPE[0]//8), point[1] % (MP_INPUT_SHAPE[1]//8)]),
        points
    )
    binsRowColumn, indices = np.unique(binsRowColumn, return_index=True, axis=0)
    
    return points[indices]




    

def recordGenerator(imgFilepaths, ptsFilepaths):
    for imgPath, ptPath in zip(imgFilepaths, ptsFilepaths): 
        image = open(imgPath, "rb").read()
        points = tf.convert_to_tensor(np.load(ptPath), dtype=tf.float32)
        # preprocess points to remove duplicates (both coordinates wise and bins wise) and decrement 120 and 160 points by 1
        points = preprocessPoints(points)
        print("I am ahere")
        bins, binsBinaryMask = generateBins(points)
        sys.exit(0)
        
        oldBins = calculateBins(points)
        
        for oldBin, point in zip(oldBins, points):
            np.testing.assert_equal(
                bins[int(point[0] % (MP_INPUT_SHAPE[0] // 8)), int(point[1] % (MP_INPUT_SHAPE[1] // 8))],
                oldBin,
                err_msg=f"Old Bins: {oldBins} || Calculated Bins: {bins} || Points: {points}"
            )
        
        indices = tf.sparse.from_dense(binsBinaryMask).indices[:, 1:-1]
        sortedPoints = points[np.lexsort((points[:, 1], points[:, 0]))]
        np.testing.assert_equal(
            sortedPoints, 
            indices, 
            err_msg=f"Points: {sortedPoints} \n Calculated Points: {indices}"
        )
        
        # Serialize tensors to byte-strings 
        serializedPoints = tf.io.serialize_tensor(points)
        serializedBins = tf.io.serialize_tensor(bins)
        
        # Create Example Message
        message = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                    "points": tf.train.Feature(bytes_list=tf.train.BytesList(value=[serializedPoints.numpy()])),
                    "bins": tf.train.Feature(bytes_list=tf.train.BytesList(value=[serializedBins.numpy()]))
                }
            )
        ).SerializeToString()
        
        yield message
        
        
def createTFRecord(filename):
    global set, generator, setDir, recordNo, maxFileSize
    with tf.io.TFRecordWriter(str(setDir / filename)) as writer:
        recordSize = 0 # MBs
        while recordSize <= maxFileSize:
            try:
                message = generator.__next__()
            except StopIteration:
                print(f"Finished Recording {set.title()} Set")
                return None
            writer.write(message)
            #recordSize += tf.strings.length(message) / (1024*1024)
            recordSize += tf.strings.length(message) / (1024*1024)

        recordNo += 1
        filename = f"{set}_record_no_{recordNo}.tfrecord"
        createTFRecord(filename)



if __name__ == "__main__":
        
    # Create directory to save tfrecords
    tfrecordsDir = Path("dataset/tfrecords")
    if not tfrecordsDir.exists():
        tfrecordsDir.mkdir()
        
    for set in dirs.keys():
        print(f"Recording {set.title()} Set")
        # Create directory for this set
        setDir = tfrecordsDir / set
        '''if setDir.exists():
            print(f"{set} directory already exists")
            continue
        setDir.mkdir()'''
        
        generator = recordGenerator(dirs[set][0], dirs[set][1])
        
        
        maxFileSize = dirs[set][2] # MBs
        recordNo = 1
        filename = f"{set}_record_no_{recordNo}.tfrecord"
        
        createTFRecord(filename)