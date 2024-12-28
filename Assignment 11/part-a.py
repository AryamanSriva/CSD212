import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def process_image():
    # Load and process image
    img = cv.imread('A.png', cv.IMREAD_GRAYSCALE)
    kernel_sz1 = np.ones(1)
    
    # Change Data type of image
    img = cv.filter2D(img, cv.CV_16U, kernel_sz1)
    
    # Threshold the image
    thresh = 127
    img = cv.threshold(img, thresh, 255, cv.THRESH_BINARY)[1]
    
    # Display results
    print('Data_type is {}', img.dtype)
    print('Image Size is {}'.format(img.shape))
    print('Image min and max are ({}, {})'.format(img.min(), img.max()))
    
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return img

if __name__ == "__main__":
    process_image()
