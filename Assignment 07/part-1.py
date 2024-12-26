import numpy as np
import matplotlib.pyplot as plt
from common_utils import create_base_signal, initialize_coefficients

def sample_signal(T):
    """Sample the signal at 2π/T intervals"""
    # Initialize coefficients
    coeffc, coeffs = initialize_coefficients(T)
    
    # Create vectorized function
    f_vec = create_base_signal(T, coeffc, coeffs)
    
    # Calculate sampling interval
    sampling_interval = 2 * np.pi / T
    
    # Generate sampled points
    x_sampled = np.arange(0, T, sampling_interval)
    y_sampled = f_vec(x_sampled)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.bar(x_sampled, y_sampled, width=sampling_interval - 0.05, align='edge', edgecolor='black')
    plt.plot(x_sampled, y_sampled, 'b-', linewidth=1)
    plt.title("Signal Sampled at 2𝜋/𝑇 Interval")
    plt.xlabel("Time")
    plt.ylabel("Signal Value")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    T = 10  # Example value
    sample_signal(T)
