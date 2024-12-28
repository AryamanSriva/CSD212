import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def extract_middle_bone():
    # Read the image
    img_B = cv.imread('B.tif', cv.IMREAD_GRAYSCALE)
    
    # Threshold to isolate the sticks
    _, thresh_B = cv.threshold(img_B, 180, 255, cv.THRESH_BINARY)
    
    # Create a large vertical structuring element
    SE_B_large = cv.getStructuringElement(cv.MORPH_RECT, (8, 19))
    
    # Erode using the large structuring element
    img_B_eroded = cv.erode(thresh_B, SE_B_large, iterations=1)
    
    # Create a smaller structuring element
    SE_B_small = cv.getStructuringElement(cv.MORPH_RECT, (8, 30))
    
    # Dilate the result to restore the curved shape
    img_B_dilated = cv.dilate(img_B_eroded, SE_B_small, iterations=1)
    
    # Perform a bitwise AND operation between the original and processed image
    img_final_B = cv.bitwise_and(img_B, img_B_dilated)
    
    # Find contours in the image
    contours_B, _ = cv.findContours(img_final_B, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by x-coordinate (L to R)
    contours_B = sorted(list(contours_B), key=lambda c: cv.boundingRect(c)[0])
    
    # Select the middle contour
    middle_contour_B = contours_B[len(contours_B) // 2]
    
    # Create an empty mask for the middle contour
    mask_B = np.zeros_like(img_B)
    
    # Draw the middle contour on the mask
    cv.drawContours(mask_B, [middle_contour_B], -1, (255), thickness=cv.FILLED)
    
    # Bitwise AND the mask with the original image
    img_final_B = cv.bitwise_and(img_B, mask_B)
    
    # Display the final output
    plt.figure(figsize=(5, 5))
    plt.imshow(img_final_B, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return img_final_B

if __name__ == "__main__":
    extract_middle_bone()
