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

# Plot the PMF
plt.bar(bins[:-1], pmf, color='b', width=1)
plt.title('PMF')
plt.xlabel('Pixel Intensity')
plt.ylabel('Probability')
plt.show()
