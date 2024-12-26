import numpy as np
import matplotlib.pyplot as plt
from common_utils import create_base_signal, initialize_coefficients

def recreate_signal(T):
    """Recreate the original signal using Fourier transform"""
    # Initialize coefficients
    coeffc, coeffs = initialize_coefficients(T)
    
    # Create vectorized function
    f_vec = create_base_signal(T, coeffc, coeffs)
    
    # Generate original signal
    dx = 0.0001
    x = np.arange(0, T, dx)
    y = f_vec(x)
    
    # Perform Fourier transform and reconstruction
    fourier_coeff = np.fft.fft(y)
    recreated_signal = np.fft.ifft(fourier_coeff).real
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'r-', label='Original Signal')
    plt.plot(x, recreated_signal, 'b-', label='Recreated Signal')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Original and Recreated Signals')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    T = 10  # Example value
    recreate_signal(T)
