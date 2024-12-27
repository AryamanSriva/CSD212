import numpy as np
from matplotlib import pyplot as plt

def apply_lowpass_filter(img_d, H_low):
    """
    Apply low pass filter to shifted FFT image
    
    Parameters:
    img_d (ndarray): Shifted FFT image
    H_low (ndarray): Low pass filter mask
    
    Returns:
    ndarray: Filtered image
    """
    fft_low_shifted = img_d * H_low
    fft_low = np.fft.ifftshift(fft_low_shifted)
    img_low = np.fft.ifft2(fft_low)
    
    print('Data_type is', img_low.dtype)
    print('Image Size is {}'.format(img_low.shape))
    print('Image min and max are ({}, {})'.format(img_low.min(), img_low.max()))
    
    img_low = img_low.real
    
    print('Data_type is', img_low.dtype)
    print('Image Size is {}'.format(img_low.shape))
    print('Image min and max are ({}, {})'.format(img_low.min(), img_low.max()))
    
    plt.figure(figsize=(5,5))
    plt.imshow(img_low, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return img_low

if __name__ == "__main__":
    from part_a import load_and_display_image
    from part_d import apply_fft_shift
    from part_f import create_meshgrid
    from part_g import create_lowpass_filter
    
    img_path = 'ckt-board-orig.tif'
    img_a0, img_a = load_and_display_image(img_path)
    img_d, _ = apply_fft_shift(img_a)
    _, _, dist = create_meshgrid(img_a0.shape)
    H_low = create_lowpass_filter(dist)
    img_low = apply_lowpass_filter(img_d, H_low)
