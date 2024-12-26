import numpy as np
import matplotlib.pyplot as plt

def create_sinc_function(frequency_parameter):
    """Create a sinc function with frequency T/2π"""
    def custom_sinc(x, T):
        # Handle the case where x is zero
        x = np.where(x == 0, 1e-10, x)
        return np.sin(np.pi * x * T) / (np.pi * x * T)
    
    x_values = np.linspace(-4 * frequency_parameter, 4 * frequency_parameter, 1000)
    y_values = custom_sinc(x_values, frequency_parameter)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values)
    plt.xlabel("x")
    plt.ylabel("sinc(x)")
    plt.title("Custom Sinc Function with Frequency T/2π")
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    frequency_parameter = 2  # Example value
    create_sinc_function(frequency_parameter)
