
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt

threshold_value = 228

# Part C - Convolve 'news-msr-2' with 'AI' and threshold the output
img_path_a = '/content/news-msr-2.png'
img_path_b = '/content/AI.png'

img0_a = cv.imread(img_path_a)
img0_b = cv.imread(img_path_b)

if img0_a is not None and img0_b is not None:
    print('Images read successfully for Part C.')

    img_c_a = cv.cvtColor(img0_a, cv.COLOR_BGR2GRAY)
    img_c_b = cv.cvtColor(img0_b, cv.COLOR_BGR2GRAY)

    img_c_a = tf.convert_to_tensor(img_c_a, dtype=tf.uint8)
    img_c_b = tf.convert_to_tensor(img_c_b, dtype=tf.uint8)

    img_c_a = tf.cast(img_c_a, tf.float32)
    img_c_b = tf.cast(img_c_b, tf.float32)

    img_c_a = tf.reshape(img_c_a, (1, img_c_a.shape[0], img_c_a.shape[1], 1))
    img_c_b = tf.reshape(img_c_b, (img_c_b.shape[0], img_c_b.shape[1], 1, 1))

    img_c_convolved = tf.nn.conv2d(img_c_a, img_c_b, strides=[1, 1, 1, 1], padding='SAME')

    threshold_c_value = tf.reduce_max(img_c_convolved) - 25
    img_c_thresholded = tf.where(img_c_convolved <= threshold_c_value, 0, 255)

    print('Image Size for Part C is', img_c_convolved.shape)
    print('Threshold Value for Part C is', threshold_c_value)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(tf.squeeze(img_c_convolved), cmap='gray')
    plt.title('Convolved Image (Part C)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(tf.squeeze(img_c_thresholded), cmap='gray')
    plt.title('Thresholded Image (Part C)')
    plt.axis('off')

    plt.show()

else:
    print('Error: Images not loaded for Part C.')
