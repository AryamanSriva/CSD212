import numpy as np
import matplotlib.pyplot as plt

def create_base_signal(T, coeffc, coeffs):
    """Create the base signal using given coefficients"""
    f = lambda x: np.sum([coeffc[i]*np.cos((i*x*2*np.pi)/T)+coeffs[i]*np.sin((i*x*2*np.pi)/T) for i in range(T)])
    return np.vectorize(f)

def initialize_coefficients(T):
    """Initialize random coefficients for cosine and sine"""
    return np.random.randn(T), np.random.randn(T)
