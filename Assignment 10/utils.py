import numpy as np
from matplotlib import pyplot as plt
import imageio.v2 as imageio

def RMSD(img1, img2):
    out1 = np.sqrt(np.power(img1-img2, 2).sum()/np.prod(img1.shape))
    return out1

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def make_noisy(noise_typ, image):
    if noise_typ == "gauss":  # Gaussian Noise
        row, col = image.shape
        mean = 0
        intensity_range = image.max()-image.min()
        sigma = 0.1*intensity_range    ###### 10% of Gaussian NOISE
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":  # Salt and Pepper Noise
        row, col = image.shape
        s_vs_p = 0.5
        amount = 0.7
        noisy = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = tuple([np.random.randint(0, i - 1, int(num_salt))
                     for i in image.shape])
        noisy[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = tuple([np.random.randint(0, i - 1, int(num_pepper))
                     for i in image.shape])
        noisy[coords] = 0
        return noisy

def load_image(path):
    img = imageio.imread(path)
    img_gray = rgb2gray(img.astype(np.float32))
    return img_gray
