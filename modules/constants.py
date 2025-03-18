import tensorflow as tf

MP_INPUT_SHAPE = [120, 160, 1]  # MagicPoint input shape
MP_BATCH_SIZE = 32
eta = 1
detection_confidences = tf.range(0.50, 1, 0.05)

train_n_samples = 90000
steps_per_epoch = 100
epochs = int(tf.floor(train_n_samples / (MP_BATCH_SIZE * steps_per_epoch)))


# Algorithm to find the following numbers
# 1. total_batches = divide train_n_samples / batch_size
# " Need to find a combination of epochs and steps_per_epochs that can fully divide total_batches"
# Divided total_batches / 4 then asked GPT to find all the divisors of the result and chose 37
epochs = 10
steps_per_epoch = 100



log_dir = "log"