import numpy as np
from matplotlib import pyplot as plt

def apply_log_transform(img_b_mod):
    """
    Apply log transform to FFT magnitude
    
    Parameters:
    img_b_mod (ndarray): Magnitude of FFT
    
    Returns:
    ndarray: Log-transformed and normalized image
    """
    min_lb = np.log2(1 + img_b_mod.min())
    max_lb = np.log2(1 + img_b_mod.max())
    
    img_b_lm = np.log2(1 + img_b_mod)
    img_c = (img_b_lm - min_lb) / (max_lb - min_lb)
    img_c = 255.0 * img_c
    
    print('lm-FFT Data_type is {}'.format(img_c.dtype))
    print('mod-FFT Size is {}'.format(img_c.shape))
    print('mod-FFT min and max are ({}, {})'.format(img_c.min(), img_c.max()))
    
    plt.figure(figsize=(3,3))
    plt.imshow(img_c, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return img_c

if __name__ == "__main__":
    from part_a import load_and_display_image
    from part_b import apply_dft
    img_path = 'ckt-board-orig.tif'
    _, img_a = load_and_display_image(img_path)
    _, img_b_mod = apply_dft(img_a)
    img_c = apply_log_transform(img_b_mod)
