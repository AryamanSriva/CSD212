import tensorflow as tf
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

# Read image using TensorFlow
img_path = 'skeleton_orig.tif'
img = imageio.imread(img_path)
img = tf.convert_to_tensor(img, dtype=tf.float32)

# Define kernel
kernel_sz1 = tf.ones((1, 1), dtype=tf.float32)

# Change data type of image using TensorFlow
img_a = tf.cast(img, tf.float32)  # Convert image to float32
img_a = tf.nn.conv2d(tf.reshape(img_a, (1,img_a.shape[0],img_a.shape[1],1)),
                     tf.reshape(kernel_sz1,
                      (kernel_sz1.shape[0],kernel_sz1.shape[1],1,1)),
                      strides=[1, 1, 1, 1], padding='SAME')

# Print information about the image
print('Original Data_type is', img.dtype)
print('Kernel_sz1 =', kernel_sz1)
print('Data_type is', img_a.dtype)
print('Image Size is', img_a.shape)
print('Image min and max are ({}, {})'.format(tf.reduce_min(img_a),
                                              tf.reduce_max(img_a)))

# Display the image
plt.figure(figsize=(10,10))
plt.imshow(tf.squeeze(img_a), cmap='gray')
plt.axis('off')
plt.show()
