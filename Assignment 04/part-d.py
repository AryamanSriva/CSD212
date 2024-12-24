import tensorflow as tf
import matplotlib.pyplot as plt
from part_a import img_a

# Define Sobel kernels
sobel_x = tf.constant([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=tf.float32)
sobel_y = tf.constant([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]], dtype=tf.float32)

# Apply Sobel operators
gradient_x = tf.nn.conv2d(tf.expand_dims(img_a, axis=0),
                       tf.expand_dims(tf.expand_dims(sobel_x, axis=-1), axis=-1),
                       strides=[1, 1, 1, 1],
                       padding='SAME')
gradient_y = tf.nn.conv2d(tf.expand_dims(img_a, axis=0),
                       tf.expand_dims(tf.expand_dims(sobel_y, axis=-1), axis=-1),
                       strides=[1, 1, 1, 1],
                       padding='SAME')

# Calculate gradient magnitude
sobel_gradient = tf.sqrt(tf.square(gradient_x) + tf.square(gradient_y))

# Print information
print('Sobel Gradient Data_type is', sobel_gradient.dtype)
print('Sobel Gradient Size is', sobel_gradient.shape)
print('Sobel Gradient min and max are ({}, {})'.format(tf.reduce_min(sobel_gradient),
                                                   tf.reduce_max(sobel_gradient)))

# Display result
plt.figure(figsize=(10, 10))
plt.imshow(tf.squeeze(sobel_gradient), cmap='gray')
plt.title('Sobel Gradient')
plt.axis('off')
plt.show()
