
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt

# Part E - Convolve the output with the reflection of 'AI' (around origin)
img_path_a = '/content/news-msr-2.png'
img_path_b = '/content/AI.png'

img0_a = cv.imread(img_path_a)
img0_b = cv.imread(img_path_b)

if img0_a is not None and img0_b is not None:
    print('Images read successfully for Part E.')

    img_c_a = cv.cvtColor(img0_a, cv.COLOR_BGR2GRAY)
    img_c_b = cv.cvtColor(img0_b, cv.COLOR_BGR2GRAY)

    img_c_a = tf.convert_to_tensor(img_c_a, dtype=tf.uint8)
    img_c_b = tf.convert_to_tensor(img_c_b, dtype=tf.uint8)

    img_c_a = tf.cast(img_c_a, tf.float32)
    img_c_b = tf.cast(img_c_b, tf.float32)

    img_c_a = tf.reshape(img_c_a, (1, img_c_a.shape[0], img_c_a.shape[1], 1))
    img_c_b_reflected = tf.reverse(tf.reshape(img_c_b, (img_c_b.shape[0], img_c_b.shape[1], 1, 1)), axis=[0, 1])

    img_c_convolved = tf.nn.conv2d(img_c_a, img_c_b_reflected, strides=[1, 1, 1, 1], padding='SAME')

    print('Image Size for Part E is', img_c_convolved.shape)

    plt.figure(figsize=(5, 5))
    plt.imshow(tf.squeeze(img_c_convolved), cmap='gray')
    plt.title('Convolved Image with Reflected AI (Part E)')
    plt.axis('off')

    plt.show()

else:
    print('Error: Images not loaded for Part E.')
