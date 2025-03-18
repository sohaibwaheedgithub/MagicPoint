import sys
import keras
from glob import glob
import tensorflow as tf
from data_preparation import DataPreparation
from constants import MP_BATCH_SIZE, MP_INPUT_SHAPE



# Convolutional Block For Shared Encoder
class SEConvBlock(keras.layers.Layer):
    def __init__(self, filters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv2d_1 = keras.layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")  
        self.batchNorm_1 = keras.layers.BatchNormalization()
        self.conv2d_2 = keras.layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")
        self.batchNorm_2 = keras.layers.BatchNormalization()
        
        
    def call(self, input):
        return self.batchNorm_2(
            self.conv2d_2(
                self.batchNorm_1(
                    self.conv2d_1(input)
                )
            )
        )
        

# Shared Encoder
class SharedEncoder(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxPool = keras.layers.MaxPool2D()
        self.SEConvBlock_1 = SEConvBlock(64)
        self.SEConvBlock_2 = SEConvBlock(64)
        self.SEConvBlock_3 = SEConvBlock(128)
        self.SEConvBlock_4 = SEConvBlock(128)
        
        
    
    def call(self, input):
        return self.SEConvBlock_4(
            self.maxPool(
                self.SEConvBlock_3(
                    self.maxPool(
                        self.SEConvBlock_2(
                            self.maxPool(
                                self.SEConvBlock_1(input)
                            )
                        )
                    )
                )
            )
        )
        

# Decoder Head For Both Interest Point And Descriptor Decoders
class DecoderHead(keras.layers.Layer):
    def __init__(self, filters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv2d = keras.layers.Conv2D(256, 3, padding="same", activation="relu", kernel_initializer="he_normal")
        self.batchNorm = keras.layers.BatchNormalization()
        self.bottleNeckLayer = keras.layers.Conv2D(filters, 1, padding="same", kernel_initializer="he_normal")
        
        
    def call(self, input):
        return self.bottleNeckLayer(
            self.batchNorm(
                self.conv2d(input)
            )
        )




# Interest Point Decoder
# Defining a separate layer for softmax operation so that Decoder Head can be used for both Decoders
class InterestPointDecoder(DecoderHead):
    def __init__(self, filters, *args, **kwargs):
        super().__init__(filters, *args, **kwargs)
    
    def call(self, input):
        return tf.nn.softmax(super().call(input))
    
        
        
# Postprocessing Layer For InterestPointDecoder to reshape outputs
class IPDPostProcessor(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable = False
        
    def call(self, inputs):
        return tf.nn.depth_to_space(inputs[..., :-1], block_size=8)
    
        
        
        
# Defining this class to overwrite the train_step method        
class MagicPoint(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
    def train_step(self, data):
        images, points, bins = data

        # Sample Weights Calculation
        n_points = tf.reduce_sum(tf.where(bins != 64, 1, 0), axis=[1, 2])
        n_points = tf.where(n_points == 0, 300, n_points)

        height, width = MP_INPUT_SHAPE[0] // 8, MP_INPUT_SHAPE[1] // 8
        n_npoints = tf.subtract(height * width, n_points)

        pointsWeights = tf.divide(n_npoints, n_points)

        n_npointsWeights = (1 / (pointsWeights + 1))[..., tf.newaxis, tf.newaxis]
        n_npointsWeights = tf.broadcast_to(n_npointsWeights, shape=[MP_BATCH_SIZE, height, width])

        pointsWeights = (pointsWeights / (pointsWeights + 1))[..., tf.newaxis, tf.newaxis]
        pointsWeights = tf.broadcast_to(pointsWeights, shape=[MP_BATCH_SIZE, height, width])

        sample_weights = tf.where(bins != 64, pointsWeights, n_npointsWeights)

        with tf.GradientTape() as tape:
            output = self(images, training=True)
            loss = self.compute_loss(
                y=bins,
                y_pred=output["interestPointDecoderOutput"],
                sample_weight=sample_weights,
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(points, output["finalOutput"])

        # Return gradients every 450 steps
        result = {m.name: m.result() for m in self.metrics}
        
        return result
    
 
    def test_step(self, data):
        images, points, bins = data
        
        n_points = tf.reduce_sum(tf.where(bins!=64, 1, 0), axis=[1, 2])
        n_points = tf.where(n_points==0, 300, n_points)
        
        height, width = MP_INPUT_SHAPE[0]//8, MP_INPUT_SHAPE[1]//8
        n_npoints = tf.subtract(height*width, n_points)

        pointsWeights = tf.divide(n_npoints, n_points)

        n_npointsWeights = (1 / (pointsWeights + 1))[..., tf.newaxis, tf.newaxis]
        n_npointsWeights = tf.broadcast_to(n_npointsWeights, shape=[MP_BATCH_SIZE, height, width])

        pointsWeights = (pointsWeights / (pointsWeights + 1))[..., tf.newaxis, tf.newaxis]
        pointsWeights = tf.broadcast_to(pointsWeights, shape=[MP_BATCH_SIZE, height, width])

        sample_weights = tf.where(bins!=64, pointsWeights, n_npointsWeights)
        
        
        output = self(images, training=False)
        loss = self.compute_loss(y=bins, y_pred=output["interestPointDecoderOutput"], sample_weight=sample_weights)
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(points, output["finalOutput"])
            
        return {m.name: m.result() for m in self.metrics}
        
        
            
        
        
# Building MagicPoint using Functional API
def buildMagicPoint(input_shape):
    input = keras.layers.Input(input_shape)
    # I am not adapting the norm layer to the dataset due to memory issues, so I have separalty calculated the mean and variance
    # over the entire dataset and setting it manually as layer's attribute
    normalizedInput = keras.layers.Normalization(axis=-1, mean=126.36381, variance=3035.8782)(input)
    sharedEncoder = SharedEncoder()(normalizedInput)
    interestPointDecoder = InterestPointDecoder(65)(sharedEncoder)
    iPDPostProcessor = IPDPostProcessor()(interestPointDecoder)
    return MagicPoint(
        inputs=[input],
        outputs={
            "sharedEncoder": sharedEncoder,
            "interestPointDecoderOutput": interestPointDecoder,
            "finalOutput": iPDPostProcessor
        }
    )
    



        
        
        
# Function to refrom shape with categorical array of bins e.g: [15, 20, 64] -> [120, 160, 1]
def reformImage(categoricalBins):
    tensor = tf.transpose(categoricalBins, perm=[0, 3, 1, 2])  # Shape becomes [64, 15, 20]
    # Step 2: Reshape the tensor to combine grids. 
    # 64 = 8 * 8, so reshape it to [8, 8, 15, 20] where each slice is a 15x20 grid.
    tensor = tf.reshape(tensor, [MP_BATCH_SIZE, 8, 8, MP_INPUT_SHAPE[0]//8, MP_INPUT_SHAPE[1]//8])
    # Step 3: Permute the axes to align the grids into a larger grid
    # This will make the shape [15*8, 20*8] = [120, 160]
    tensor = tf.transpose(tensor, perm=[0, 1, 3, 2, 4])
    reformedImage = tf.reshape(tensor, [MP_BATCH_SIZE] + MP_INPUT_SHAPE)
    return reformedImage
    
        

if __name__ == "__main__":
    # Dataset preparation
    dataPreparation = DataPreparation(MP_BATCH_SIZE)
    filenames = glob("dataset/tfrecords/train/*")
    dataset = dataPreparation.loadDataset(filenames)
    iterator = dataset.__iter__()
    images, points, bins = iterator.__next__()
    
    # Model initialization
    magicPoint = buildMagicPoint(MP_INPUT_SHAPE)
    
    # Model calling
    modelOutputs = magicPoint(images)
    print(magicPoint.summary())
    """print(modelOutputs["interestPointDecoderOutput"].shape)
    print(modelOutputs["finalOutput"].shape)"""
    
    # Verifying if softmax has been correctly applied channel wise
    '''x = tf.math.round(tf.reduce_sum(modelOutputs["interestPointDecoderOutput"], axis=-1))
    y = tf.ones_like(x)
    print(tf.reduce_all(tf.equal(x, y)).numpy())'''