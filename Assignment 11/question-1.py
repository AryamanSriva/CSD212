import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def remove_T_from_UTK():
    # Read the images
    img_UTK = cv.imread('UTK.tif', cv.IMREAD_GRAYSCALE)
    img_T = cv.imread('T.tif', cv.IMREAD_GRAYSCALE)
    
    # Threshold for the letter 'T'
    _, thresh_T = cv.threshold(img_T, 127, 255, cv.THRESH_BINARY)
    structuring_element_T = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    img_T_eroded = cv.erode(thresh_T, structuring_element_T, iterations=1)
    img_T_dilated = cv.dilate(img_T_eroded, structuring_element_T, iterations=2)
    
    # Resize 'T' to match the size of 'UTK'
    img_T_resized = cv.resize(img_T_dilated, (img_UTK.shape[1], img_UTK.shape[0]))
    
    # Shift 'T' to the right
    shift_pixels = 2
    img_T_shifted = np.roll(img_T_resized, shift_pixels, axis=1)
    
    # Perform XOR operation to remove 'T' from 'UTK'
    img_xor = cv.bitwise_xor(img_UTK, img_T_shifted)
    
    # Perform AND operation to ensure only the 'T' region is removed
    img_final = cv.bitwise_and(img_xor, img_UTK)
    
    # Display the final output
    plt.figure(figsize=(5, 5))
    plt.imshow(img_final, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return img_final

if __name__ == "__main__":
    remove_T_from_UTK()
