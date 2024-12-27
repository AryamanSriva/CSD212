import numpy as np
from matplotlib import pyplot as plt

def create_lowpass_filter(dist, radius=80):
    """
    Create ideal low pass filter
    
    Parameters:
    dist (ndarray): Distance array from meshgrid
    radius (int): Radius for the low pass filter
    
    Returns:
    ndarray: Low pass filter mask
    """
    H_low = (dist <= radius)
    
    plt.figure(figsize=(3,3))
    plt.imshow(H_low, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return H_low

if __name__ == "__main__":
    from part_a import load_and_display_image
    from part_f import create_meshgrid
    img_path = 'ckt-board-orig.tif'
    img_a0, _ = load_and_display_image(img_path)
    _, _, dist = create_meshgrid(img_a0.shape)
    H_low = create_lowpass_filter(dist)
