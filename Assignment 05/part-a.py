import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

def load_xray_image(img_path):
    """
    Load and preprocess X-ray circuit image.
    
    Args:
        img_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed grayscale image
    """
    img = tf.io.read_file(img_path)
    img_a = tf.io.decode_jpeg(img)
    img_a = tf.image.rgb_to_grayscale(img_a).numpy()
    
    print('Original Data_type is {}'.format(img_a.dtype))
    print('Data_type is', img_a.dtype)
    print('Image Size is {}'.format(img_a.shape))
    print('Image min and max are ({}, {})'.format(img_a.min(), img_a.max()))
    
    return img_a

def display_image(img):
    """
    Display the image using matplotlib.
    
    Args:
        img (numpy.ndarray): Image to display
    """
    plt.figure(figsize=(8,8))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    img_path = 'xray_circuit.jpg'
    img_a = load_xray_image(img_path)
    display_image(img_a)
