import numpy as np
import matplotlib.pyplot as plt

def calculate_motion_blur_filter(shape):
    """
    Calculate the motion blur filter H(u,v)
    
    Parameters:
    shape (tuple): Shape of the image (M, N)
    
    Returns:
    numpy.ndarray: Motion blur filter H(u,v)
    """
    M, N = shape
    [vv, uu] = np.meshgrid(range(N), range(M))
    uu = uu - 0.5 * M
    vv = vv - 0.5 * N
    
    PI = np.pi
    A = 0.02
    B = 0.02
    T = 1
    
    sinc01 = np.sinc(PI * ((uu) * A + (vv) * B))
    ej = np.exp(-(1j) * PI * ((uu) * A + (vv) * B))
    
    return T * sinc01 * ej

def calculate_wiener_filter(H, K=0.000001):
    """
    Calculate the Wiener filter using the motion blur filter H(u,v)
    
    Parameters:
    H (numpy.ndarray): Motion blur filter
    K (float): Noise-to-signal ratio parameter
    
    Returns:
    numpy.ndarray: Wiener filter
    """
    Wiener_filter = np.conj(H) / (np.abs(H)**2 + K)
    
    # Visualize the filter
    plt.figure(figsize=(5, 5))
    plt.imshow(np.abs(Wiener_filter), cmap='gray')
    plt.axis('off')
    plt.title('Wiener Filter')
    plt.show()
    
    return Wiener_filter

if __name__ == "__main__":
    # Example usage with a 1024x1024 image
    shape = (1024, 1024)
    H = calculate_motion_blur_filter(shape)
    Wiener_filter = calculate_wiener_filter(H)
