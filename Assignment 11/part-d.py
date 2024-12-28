import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from part_b import create_structuring_element
from part_c import erode_image

def dilate_image():
    # Get eroded image and structuring element
    img_erode = erode_image()
    SE1 = create_structuring_element()
    
    # Perform dilation twice
    img_e_d = cv.dilate(img_erode, SE1, iterations=2)
    
    # Display results
    print('Data_type is {}', img_e_d.dtype)
    print('Image Size is {}'.format(img_e_d.shape))
    print('Image min and max are ({}, {})'.format(img_e_d.min(), img_e_d.max()))
    
    plt.figure(figsize=(10,10))
    plt.imshow(img_e_d, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return img_e_d

if __name__ == "__main__":
    dilate_image()
