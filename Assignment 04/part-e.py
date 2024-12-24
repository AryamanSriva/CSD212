import tensorflow as tf
import matplotlib.pyplot as plt
from part_d import sobel_gradient

# Define 5x5 averaging filter
avg_filter = tf.constant([[1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1]], dtype=tf.float32) / 25.0

# Apply smoothing
smoothed_sobel_gradient = tf.nn.conv2d(tf.expand_dims(sobel_gradient, axis=0),
                                   tf.expand_dims(tf.expand_dims(avg_filter, axis=-1), axis=-1),
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')

# Print information
print('Smoothed Sobel Gradient Data_type is', smoothed_sobel_gradient.dtype)
print('Smoothed Sobel Gradient Size is', smoothed_sobel_gradient.shape)
print('Smoothed Sobel Gradient min and max are ({}, {})'.format(tf.reduce_min(smoothed_sobel_gradient),
                                                            tf.reduce_max(smoothed_sobel_gradient)))

# Display result
plt.figure(figsize=(10, 10))
plt.imshow(tf.squeeze(smoothed_sobel_gradient), cmap='gray')
plt.title('Smoothed Sobel Gradient')
plt.axis('off')
plt.show()
