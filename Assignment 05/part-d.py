import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from part_a import load_xray_image, display_image

def apply_anisotropic_diffusion(img, niter=200, kappa=8, lambda_val=0.15):
    """
    Apply anisotropic diffusion filtering.
    
    Args:
        img (numpy.ndarray): Input image
        niter (int): Number of iterations
        kappa (float): Conductance coefficient
        lambda_val (float): Lambda parameter
        
    Returns:
        numpy.ndarray: Processed image
    """
    # Initialize image tensor
    img_tensor = tf.constant(img, shape=(1, img.shape[0], img.shape[1], img.shape[2]), 
                           dtype=tf.float64)
    
    # Define filters
    filter_x = tf.constant([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=tf.float64)
    filter_y = tf.constant([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=tf.float64)
    k_squared = tf.constant(kappa**2, dtype=tf.float64)
    
    # Gaussian filter for exp term
    filter_exp_term = tf.constant([
        [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902],
        [0.01330621, 0.0596343 , 0.09832033, 0.0596343 , 0.01330621],
        [0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823],
        [0.01330621, 0.0596343 , 0.09832033, 0.0596343 , 0.01330621],
        [0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902]
    ], dtype=tf.float64)
    
    result = img_tensor
    
    # Apply diffusion iterations
    for _ in range(niter):
        # Calculate gradients
        grad_x = tf.nn.conv2d(result, 
                             tf.reshape(filter_x, [3, 3, 1, 1]), 
                             strides=[1, 1, 1, 1], 
                             padding='SAME')
        grad_y = tf.nn.conv2d(result, 
                             tf.reshape(filter_y, [3, 3, 1, 1]), 
                             strides=[1, 1, 1, 1], 
                             padding='SAME')
        
        # Calculate gradient magnitude and exponential terms
        grad_mag = tf.abs(grad_x) + tf.abs(grad_y)
        grad_squared = grad_mag ** 2
        exp_term = tf.exp(-grad_squared / k_squared)
        
        # Apply filtering
        num = tf.nn.conv2d(exp_term, 
                          tf.reshape(filter_exp_term, [5, 5, 1, 1]), 
                          strides=[1, 1, 1, 1], 
                          padding='SAME')
        den = tf.nn.conv2d(exp_term, 
                          tf.reshape(filter_exp_term, [5, 5, 1, 1]), 
                          strides=[1, 1, 1, 1], 
                          padding='SAME')
        
        # Update result
        result *= num / (den + 1e-8)
    
    # Normalize and scale output
    result_min = tf.reduce_min(result)
    result_max = tf.reduce_max(result)
    result_norm = (result - result_min) / (result_max - result_min)
    result_scaled = result_norm * (img.max() - img.min()) + img.min()
    
    return tf.cast(result_scaled, dtype=tf.uint8).numpy().squeeze()

if __name__ == "__main__":
    img_path = 'xray_circuit.jpg'
    img_a = load_xray_image(img_path)
    processed_img = apply_anisotropic_diffusion(img_a)
    display_image(processed_img)
