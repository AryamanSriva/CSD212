
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv

threshold_value = 228

# Part A - Process 'news-msr-2.png'
img_path_a = '/content/news-msr-2.png'
img0_a = cv.imread(img_path_a)
img_a = cv.cvtColor(img0_a, cv.COLOR_BGR2GRAY)

img_a = tf.convert_to_tensor(img_a, dtype=tf.uint8)
kernel_sz1 = tf.ones((1, 1), dtype=tf.float32)
img_a = tf.cast(img_a, tf.float32)
img_a = tf.nn.conv2d(tf.reshape(img_a, (1, img_a.shape[0], img_a.shape[1], 1)),
                     tf.reshape(kernel_sz1, (1, 1, 1, 1)),
                     strides=[1, 1, 1, 1],
                     padding='SAME')

img_a_thresholded = tf.where(img_a <= threshold_value, 0, 255)

print('Original Data_type is', img_a.dtype)
print('Kernel_sz1 =', kernel_sz1)
print('Data_type is', img_a.dtype)
print('Image Size is', img_a.shape)
print('Image min and max are ({}, {})'.format(tf.reduce_min(img_a), tf.reduce_max(img_a)))

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(tf.squeeze(img_a), cmap='gray')
plt.title('Processed Image with 1x1 Identity Kernel (Part A)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(tf.squeeze(img_a_thresholded), cmap='gray')
plt.title('Thresholded Image (Part A)')
plt.axis('off')
