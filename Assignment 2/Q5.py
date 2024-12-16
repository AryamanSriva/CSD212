import cv2 as cv
import numpy as np
from skimage.exposure import match_histograms
from matplotlib import pyplot as plt

# Load the reference and input images
ref0 = cv.imread('penstand_bright.jpg')
ref = cv.cvtColor(ref0, cv.COLOR_BGR2GRAY)

inp0 = cv.imread('penstand_lowlight2.jpg')
inp = cv.cvtColor(inp0, cv.COLOR_BGR2GRAY)

# Perform histogram matching using match_histograms
matched = match_histograms(inp, ref)

# Calculate histograms for the input, matched, and reference images
hist_inp, bins_inp = np.histogram(inp.ravel(), 256, [0, 256])
hist_matched, bins_matched = np.histogram(matched.ravel(), 256, [0, 256])
hist_ref, bins_ref = np.histogram(ref.ravel(), 256, [0, 256])

# Display three histograms side by side
plt.figure(figsize=(20, 8))

plt.subplot(131)
plt.bar(bins_inp[:-1], hist_inp, color='b')
plt.title('Histogram of Input Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(132)
plt.bar(bins_matched[:-1], hist_matched, color='r')
plt.title('Histogram of Matched Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(133)
plt.bar(bins_ref[:-1], hist_ref, color='r')
plt.title('Histogram of Reference Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.show()
