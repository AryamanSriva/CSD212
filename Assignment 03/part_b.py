
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv

threshold_value = 228

# Part B - Process 'AI.png'
img_path_b = '/content/AI.png'
img0_b = cv.imread(img_path_b)

if img0_b is not None:
    print('Image read successfully for Part B.')
    img_b = cv.cvtColor(img0_b, cv.COLOR_BGR2GRAY)

    img_b = tf.convert_to_tensor(img_b, dtype=tf.uint8)

    plt.figure(figsize=(5, 5))
    plt.imshow(tf.squeeze(img_b), cmap='gray')
    plt.title('Original Image (Part B)')
    plt.axis('off')

    img_b = tf.cast(img_b, tf.float32)
    img_b = tf.nn.conv2d(tf.reshape(img_b, (1, img_b.shape[0], img_b.shape[1], 1)),
                         tf.reshape(tf.ones((1, 1), dtype=tf.float32), (1, 1, 1, 1)),
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    img_b_thresholded = tf.where(img_b <= threshold_value, 0, 255)

    print(img_b.shape)

    plt.figure(figsize=(5, 5))
    plt.imshow(tf.squeeze(img_b_thresholded), cmap='gray')
    plt.title('Thresholded Image (Part B)')
    plt.axis('off')

    plt.show()
else:
    print('Error: Image not loaded for Part B.')
