import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from part_a import load_xray_image, display_image

def apply_nonlinear_diffusion(img, niter=100, kappa=8, option=2, lambda_val=0.15):
    """
    Apply non-linear isotropic diffusion.
    
    Args:
        img (numpy.ndarray): Input image
        niter (int): Number of iterations
        kappa (float): Conductance coefficient
        option (int): Diffusion equation type (1 or 2)
        lambda_val (float): Lambda parameter for diffusion
        
    Returns:
        numpy.ndarray: Processed image
    """
    # Initialize variables
    img_tensor = tf.constant(img, shape=(1, img.shape[0], img.shape[1], img.shape[2]), dtype=tf.float64)
    
    # Create Sobel filters
    sobel_y = tf.constant([[0,-1,0], [0, 0, 0], [0, 1, 0]], dtype=tf.float64)
    sobel_x = tf.constant([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=tf.float64)
    
    imgout = img_tensor
    
    # Iterate diffusion process
    for _ in range(niter):
        grad_x = tf.nn.conv2d(imgout, 
                             tf.expand_dims(tf.expand_dims(sobel_x, axis=-1), axis=-1), 
                             strides=[1, 1, 1, 1], 
                             padding='SAME')
        grad_y = tf.nn.conv2d(imgout, 
                             tf.expand_dims(tf.expand_dims(sobel_y, axis=-1), axis=-1), 
                             strides=[1, 1, 1, 1], 
                             padding='SAME')
        
        # Compute conductance coefficient
        if option == 1:
            c = tf.exp(-(grad_x**2 + grad_y**2) / kappa**2)
        else:
            c = 1 / (1 + (grad_x**2 + grad_y**2) / kappa**2)
        
        # Update image
        imgout = imgout + lambda_val * (
            tf.nn.conv2d(c * grad_x, 
                        tf.expand_dims(tf.expand_dims(sobel_x, axis=-1), axis=-1), 
                        strides=[1, 1, 1, 1], 
                        padding='SAME') + 
            tf.nn.conv2d(c * grad_y, 
                        tf.expand_dims(tf.expand_dims(sobel_y, axis=-1), axis=-1), 
                        strides=[1, 1, 1, 1], 
                        padding='SAME')
        )
    
    # Normalize output
    imgout_min = tf.reduce_min(imgout)
    imgout_max = tf.reduce_max(imgout)
    img_norm = (imgout - imgout_min) / (imgout_max - imgout_min)
    img_scaled = img_norm * (img.max() - img.min()) + img.min()
    
    return tf.cast(img_scaled, dtype=tf.uint8).numpy().squeeze()

if __name__ == "__main__":
    img_path = 'xray_circuit.jpg'
    img_a = load_xray_image(img_path)
    processed_img = apply_nonlinear_diffusion(img_a)
    display_image(processed_img)
