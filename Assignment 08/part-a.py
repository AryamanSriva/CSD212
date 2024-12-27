import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import imageio.v2 as imageio

def load_and_display_image(img_path):
    """
    Load and display an image from the given path
    
    Parameters:
    img_path (str): Path to the image file
    
    Returns:
    tuple: Original image array (uint8), converted image array (float32)
    """
    img_a0 = imageio.imread(img_path)
    img_a = img_a0.astype(np.float32)
    
    print('Original Data_type is {}'.format(img_a0.dtype))
    print('Data_type is', img_a.dtype)
    print('Image Size is {}'.format(img_a.shape))
    print('Image min and max are ({}, {})'.format(img_a.min(), img_a.max()))
    
    plt.figure(figsize=(5,5))
    plt.imshow(img_a, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return img_a0, img_a

if __name__ == "__main__":
    img_path = 'ckt-board-orig.tif'
    img_a0, img_a = load_and_display_image(img_path)
