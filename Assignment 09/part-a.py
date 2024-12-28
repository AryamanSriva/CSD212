import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

def show_shifted_fourier_transform(img_low_mod):
    """
    Display the shifted Fourier transform of a motion blurred image
    
    Parameters:
    img_low_mod (numpy.ndarray): Motion blurred input image
    """
    # Perform Fourier transform
    img_fft = np.fft.fft2(img_low_mod)

    # Compute the Shifted Fourier Transform
    img_fft_shifted = np.fft.fftshift(img_fft)

    # Add a small constant to avoid taking log of zero or negative values
    epsilon = 1e-10
    img_fft_shifted_positive = np.abs(img_fft_shifted) + epsilon

    # Apply log transform
    img_fft_shifted_log = np.log(img_fft_shifted_positive)

    # Visualize the log-transformed image
    plt.figure(figsize=(5, 5))
    plt.imshow(img_fft_shifted_log, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return img_fft_shifted

if __name__ == "__main__":
    # Load and process original image
    img_path = 'book_image_orig.tif'
    img_a0 = imageio.imread(img_path)
    img_a = img_a0.astype(np.float32)
    
    # Padding parameters
    pad_sz = [344, 344]
    img_a = np.pad(img_a, (pad_sz[0], pad_sz[0]))
    
    # Show shifted Fourier transform
    img_fft_shifted = show_shifted_fourier_transform(img_a)
