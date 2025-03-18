import tensorflow as tf



batch_size = 2
# Original tensor
#firstTensor = tf.experimental.numpy.random.randint(1, 5, [batch_size, 3, 2, 4])
firstTensor = tf.random.uniform([batch_size, 15, 20, 64])

# Step 1: Transpose the tensor to bring the 64 channels in line for easier reshaping
tensor = tf.transpose(firstTensor, perm=[0, 3, 1, 2])  # Shape becomes [64, 15, 20]

# Step 2: Reshape the tensor to combine grids. 
# 64 = 8 * 8, so reshape it to [8, 8, 15, 20] where each slice is a 15x20 grid.
tensor = tf.reshape(tensor, [batch_size, 8, 8, 15, 20])

# Step 3: Permute the axes to align the grids into a larger grid
# This will make the shape [15*8, 20*8] = [120, 160]
tensor = tf.transpose(tensor, perm=[0, 1, 3, 2, 4])
tensor = tf.reshape(tensor, [batch_size, 120, 160])


print(firstTensor[0, :1, :1, 1])
print(tensor[0, :1, 20:21])