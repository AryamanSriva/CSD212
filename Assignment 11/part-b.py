import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def create_structuring_element():
    # Create 20x20 rectangular structuring element
    SE1 = cv.getStructuringElement(cv.MORPH_RECT, (20,20))
    
    # Display results
    print('Data_type is {}', SE1.dtype)
    print('Image Size is {}'.format(SE1.shape))
    print('Image min and max are ({}, {})'.format(SE1.min(), SE1.max()))
    
    plt.figure(figsize=(2,2))
    plt.imshow(SE1, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return SE1

if __name__ == "__main__":
    create_structuring_element()
