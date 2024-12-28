import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from part_b import create_structuring_element
from part_d import dilate_image

def process_image_e():
    # Get dilated image and first structuring element
    img_e_d = dilate_image()
    SE1 = create_structuring_element()
    
    # Erode three times
    img_e_d_d_e = cv.erode(img_e_d, SE1, iterations=3)
    
    # Create second structuring element and dilate 8 times
    SE2 = cv.getStructuringElement(cv.MORPH_RECT, (4,4))
    img_e_d_d_e = cv.dilate(img_e_d_d_e, SE2, iterations=8)
    
    # Display results
    print('Data_type is {}', img_e_d_d_e.dtype)
    print('Image Size is {}'.format(img_e_d_d_e.shape))
    print('Image min and max are ({}, {})'.format(img_e_d_d_e.min(), img_e_d_d_e.max()))
    
    plt.figure(figsize=(10,10))
    plt.imshow(img_e_d_d_e, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return img_e_d_d_e

if __name__ == "__main__":
    process_image_e()
