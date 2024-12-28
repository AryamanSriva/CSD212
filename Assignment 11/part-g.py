import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from part_a import process_image
from part_e import process_image_e

def shift_and_xor():
    # Get original and processed images
    img = process_image()
    img_e_d_d_e = process_image_e()
    
    # Create shifting structuring element
    SE3 = np.zeros((5,5), dtype=np.uint8)
    SE3[2:,2:] = 1
    
    # Display structuring element
    plt.figure(figsize=(2,2))
    plt.imshow(SE3, cmap='gray')
    plt.axis('off')
    plt.show()
    
    # Perform dilation with shifting
    img_e_d_d_e1 = cv.dilate(img_e_d_d_e, SE3, iterations=7)
    
    # Display intermediate result
    print('Data_type is {}', img_e_d_d_e1.dtype)
    print('Image Size is {}'.format(img_e_d_d_e1.shape))
    print('Image min and max are ({}, {})'.format(img_e_d_d_e1.min(), img_e_d_d_e1.max()))
    
    plt.figure(figsize=(10,10))
    plt.imshow(img_e_d_d_e1, cmap='gray')
    plt.axis('off')
    
    # Perform final XOR operation
    img_c = cv.bitwise_xor(img, img_e_d_d_e1)
    
    # Display final result
    print('Data_type is {}', img_c.dtype)
    print('Image Size is {}'.format(img_c.shape))
    print('Image min and max are ({}, {})'.format(img_c.min(), img_c.max()))
    
    plt.figure(figsize=(10,10))
    plt.imshow(img_c, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return img_c

if __name__ == "__main__":
    shift_and_xor()
