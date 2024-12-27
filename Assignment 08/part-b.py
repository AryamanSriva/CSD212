import numpy as np

def apply_dft(img_a):
    """
    Apply Discrete Fourier Transform to the input image
    
    Parameters:
    img_a (ndarray): Input image array
    
    Returns:
    ndarray: FFT of the input image, magnitude of FFT
    """
    img_b = np.fft.fft2(img_a)
    print('FFT Data_type is {}'.format(img_b.dtype))
    print('FFT Size is {}'.format(img_b.shape))
    print('FFT min and max are ({}, {})'.format(img_b.min(), img_b.max()))
    
    img_b_mod = np.abs(img_b)
    print('mod-FFT Data_type is {}'.format(img_b_mod.dtype))
    print('mod-FFT Size is {}'.format(img_b_mod.shape))
    print('mod-FFT min and max are ({}, {})'.format(img_b_mod.min(), img_b_mod.max()))
    print('log mod-FFT min and max are ({}, {})'.format(
        np.log2(1+img_b_mod.min()), np.log2(1+img_b_mod.max())))
    
    return img_b, img_b_mod

if __name__ == "__main__":
    from part_a import load_and_display_image
    img_path = 'ckt-board-orig.tif'
    _, img_a = load_and_display_image(img_path)
    img_b, img_b_mod = apply_dft(img_a)
