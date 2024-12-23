import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Load the image and convert it to grayscale
img0 = cv.imread('people_lowlight.jpg')
img = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)

# Display the image
plt.figure(figsize=(5,5))
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()

# Flatten the image
img_flat = img.ravel()

# Calculate the histogram
hist, bins = np.histogram(img_flat, 256, [0, 256])

# Calculate the probability mass function (PMF)
pmf = hist / img_flat.size

# Calculate the cumulative distribution function (CDF)
cdf = np.cumsum(pmf)

# Plot the CDF
plt.plot(bins[:-1], cdf, color='r')
plt.title('Cumulative Distribution Function (CDF) of Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Cumulative Probability')
plt.show()
