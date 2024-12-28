import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from part_a import process_image
from part_b import create_structuring_element

def erode_image():
    # Get image and structuring element
    img = process_image()
    SE1 = create_structuring_element()
    
    # Perform erosion
    img_erode = cv.erode(img, SE1, iterations=1)
    
    # Display results
    print('Data_type is {}', img_erode.dtype)
    print('Image Size is {}'.format(img_erode.shape))
    print('Image min and max are ({}, {})'.format(img_erode.min(), img_erode.max()))
    
    plt.figure(figsize=(10,10))
    plt.imshow(img_erode, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return img_erode

if __name__ == "__main__":
    erode_image()
