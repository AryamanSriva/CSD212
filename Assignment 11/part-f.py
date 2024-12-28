import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from part_a import process_image
from part_e import process_image_e

def xor_images():
    # Get original and processed images
    img = process_image()
    img_e_d_d_e = process_image_e()
    
    # Perform XOR operation
    img_b = cv.bitwise_xor(img, img_e_d_d_e)
    
    # Display results
    print('Data_type is {}', img_b.dtype)
    print('Image Size is {}'.format(img_b.shape))
    print('Image min and max are ({}, {})'.format(img_b.min(), img_b.max()))
    
    plt.figure(figsize=(10,10))
    plt.imshow(img_b, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return img_b

if __name__ == "__main__":
    xor_images()
