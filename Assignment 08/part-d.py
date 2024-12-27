import numpy as np
from matplotlib import pyplot as plt

def apply_fft_shift(img_a):
    """
    Apply FFT shift to the image
    
    Parameters:
    img_a (ndarray): Input image array
    
    Returns:
    tuple: Shifted FFT, log-transformed and normalized shifted FFT
    """
    img_fft_a = np.fft.fft2(img_a)
    img_d = np.fft.fftshift(img_fft_a)
    
    print('shift-FFT Data_type is {}'.format(img_d.dtype))
    print('shift-FFT Size is {}'.format(img_d.shape))
    print('shift-FFT min and max are ({}, {})'.format(img_d.min(), img_d.max()))
    
    img_d_mod = np.abs(img_d)
    min_ld = np.log2(1 + img_d_mod.min())
    max_ld = np.log2(1 + img_d_mod.max())
    
    img_d_lm = np.log2(1 + img_d_mod)
    img_e = (img_d_lm - min_ld) / (max_ld - min_ld)
    img_e = 255.0 * img_e
    
    print('shift-lm-FFT Data_type is {}'.format(img_e.dtype))
    print('shift-mod-FFT Size is {}'.format(img_e.shape))
    print('shift-mod-FFT min and max are ({}, {})'.format(img_e.min(), img_e.max()))
    
    plt.figure(figsize=(3,3))
    plt.imshow(img_e, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return img_d, img_e

if __name__ == "__main__":
    from part_a import load_and_display_image
    img_path = 'ckt-board-orig.tif'
    _, img_a = load_and_display_image(img_path)
    img_d, img_e = apply_fft_shift(img_a)
