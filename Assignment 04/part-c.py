import tensorflow as tf
import matplotlib.pyplot as plt
from part_a import img_a

# Define sharpening kernel
sharpen_kernel = tf.constant([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=tf.float32)

# Apply sharpening
sharpened_img = tf.nn.conv2d(tf.expand_dims(img_a, axis=0),
                          tf.expand_dims(tf.expand_dims(sharpen_kernel, axis=-1), axis=-1),
                          strides=[1, 1, 1, 1],
                          padding='SAME')

# Print information
print('Sharpened Image Data_type is', sharpened_img.dtype)
print('Sharpened Image Size is', sharpened_img.shape)
print('Sharpened Image min and max are ({}, {})'.format(tf.reduce_min(sharpened_img),
                                                     tf.reduce_max(sharpened_img)))

# Display result
plt.figure(figsize=(10, 10))
plt.imshow(tf.squeeze(sharpened_img), cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')
plt.show()
