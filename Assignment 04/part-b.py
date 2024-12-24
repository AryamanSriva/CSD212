import tensorflow as tf
import matplotlib.pyplot as plt
from part_a import img_a

# Define Laplacian kernel
laplacian_kernel = tf.constant([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]], dtype=tf.float32)

# Apply Laplacian filter
laplacian_img = tf.nn.conv2d(tf.expand_dims(img_a, axis=0),
                           tf.expand_dims(tf.expand_dims(laplacian_kernel, axis=-1), axis=-1),
                           strides=[1, 1, 1, 1],
                           padding='SAME')

# Print information
print('Laplacian Filtered Data_type is', laplacian_img.dtype)
print('Laplacian Filtered Image Size is', laplacian_img.shape)
print('Laplacian Filtered Image min and max are ({}, {})'.format(tf.reduce_min(laplacian_img),
                                                              tf.reduce_max(laplacian_img)))

# Display results
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(tf.squeeze(img_a), cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(tf.squeeze(laplacian_img), cmap='gray')
plt.title('Laplacian Filtered Image')
plt.axis('off')

plt.show()
