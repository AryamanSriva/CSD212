import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Load the image and convert it to grayscale
img0 = cv.imread('people_lowlight.jpg')
img = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)

# Display the original and equalized images side by side
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('Original')

# Flatten the image
img_flat = img.ravel()

# Calculate the histogram
hist, bins = np.histogram(img_flat, 256, [0, 256])

# Calculate the probability mass function (PMF)
pmf = hist / img_flat.size

# Calculate the cumulative distribution function (CDF)
cdf = np.cumsum(pmf)

# Calculate the transformation function T(r)
L = 256  # Number of intensity levels
T = np.round((L - 1) * cdf).astype('uint8')  # Corrected line

# Apply histogram equalization using T(r)
img_equalized = T[img]

# Equalized Image
plt.subplot(1, 2, 2)
plt.imshow(img_equalized, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('T(r)')

plt.show()
