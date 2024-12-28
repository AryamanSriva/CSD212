import numpy as np
import matplotlib.pyplot as plt
from part_a import show_shifted_fourier_transform
from part_b import calculate_motion_blur_filter, calculate_wiener_filter

def apply_wiener_filter(img, K=0.000001):
    """
    Apply Wiener filter to a motion blurred image and reconstruct it
    
    Parameters:
    img (numpy.ndarray): Input motion blurred image
    K (float): Noise-to-signal ratio parameter for Wiener filter
    
    Returns:
    numpy.ndarray: Reconstructed image
    """
    # Get shifted Fourier transform
    img_fft_shifted = show_shifted_fourier_transform(img)
    
    # Calculate motion blur filter
    H = calculate_motion_blur_filter(img.shape)
    
    # Calculate Wiener filter
    Wiener_filter = calculate_wiener_filter(H, K)
    
    # Apply Wiener filter
    img_fft_filtered = img_fft_shifted * Wiener_filter
    
    # Reconstruct the image
    img_reconstructed = np.fft.ifft2(np.fft.ifftshift(img_fft_filtered)).real
    
    # Display result
    plt.figure(figsize=(5, 5))
    plt.imshow(img_reconstructed, cmap='gray')
    plt.axis('off')
    plt.title('Reconstructed Image')
    plt.show()
    
    return img_reconstructed

if __name__ == "__main__":
    import imageio.v2 as imageio
    
    # Load and process original image
    img_path = 'book_image_orig.tif'
    img_a0 = imageio.imread(img_path)
    img_a = img_a0.astype(np.float32)
    
    # Padding parameters
    pad_sz = [344, 344]
    img_a = np.pad(img_a, (pad_sz[0], pad_sz[0]))
    
    # Apply Wiener filter and reconstruct
    img_reconstructed = apply_wiener_filter(img_a)
