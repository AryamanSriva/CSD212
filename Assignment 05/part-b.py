import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from part_a import load_xray_image, display_image

def create_gaussian_filter():
    """
    Create a 5x5 Gaussian filter.
    
    Returns:
        numpy.ndarray: Normalized Gaussian filter
    """
    frange = np.arange(-2,3)
    X, Y = np.meshgrid(frange, frange, indexing='ij')
    filter = np.exp(-(X**2 + Y**2)/2)
    filter /= filter.ravel().sum()
    return filter

def apply_isotropic_diffusion(img, num_iterations=10):
    """
    Apply isotropic diffusion using Gaussian filter.
    
    Args:
        img (numpy.ndarray): Input image
        num_iterations (int): Number of times to apply the filter
    
    Returns:
        numpy.ndarray: Filtered image
    """
    filter = create_gaussian_filter()
    gauss_filter = tf.constant(filter, shape=(5, 5, 1, 1))
    
    img_tensor = tf.constant(img, shape=(1, img.shape[0], img.shape[1], img.shape[2]))
    img_tensor = tf.cast(img_tensor, tf.float64)
    
    result = img_tensor
    for _ in range(num_iterations):
        result = tf.nn.conv2d(result, gauss_filter, strides=[1, 1, 1, 1], padding='SAME')
    
    return result.numpy().astype(np.uint8).squeeze()

if __name__ == "__main__":
    img_path = 'xray_circuit.jpg'
    img_a = load_xray_image(img_path)
    filtered_img = apply_isotropic_diffusion(img_a)
    display_image(filtered_img)
