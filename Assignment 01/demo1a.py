import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read image in grayscale
img = cv.imread('mammogram_digital_Xray.tif', cv.IMREAD_GRAYSCALE)

# Print image size
print('Image Size is {}'.format(img.shape))

# Calculate maximum intensity
L = np.max(img)

# Create negative image
neg_img = L - 1 - img

# Original image
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('Original Image')
plt.show()

# Negative image
plt.figure(figsize=(5, 5))
plt.imshow(neg_img, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('Negative Image')
plt.show()