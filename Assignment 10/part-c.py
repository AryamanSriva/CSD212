import numpy as np
import matplotlib.pyplot as plt
from utils import make_noisy, load_image

def plot_max_eigenvalues(image_path):
    # Parameters
    K = 290
    neighbour_size = 31
    patch_size = 7
    random_R = (neighbour_size - patch_size) // 2
    
    # Load and prepare image
    img_gray = load_image(image_path)
    noisy_img = make_noisy("gauss", img_gray)
    noisy_padded = np.pad(noisy_img, ((neighbour_size, neighbour_size), 
                                     (neighbour_size, neighbour_size)), 
                         mode='symmetric')
    
    # Initialize max eigenvalues array
    max_eigenvalues = np.zeros_like(noisy_img)
    
    # Iterate over patches
    for i_val in range(neighbour_size, noisy_padded.shape[0] - neighbour_size, patch_size):
        for j_val in range(neighbour_size, noisy_padded.shape[1] - neighbour_size, patch_size):
            # Get current patch
            patch = noisy_padded[i_val - patch_size // 2: i_val + patch_size // 2 + 1,
                               j_val - patch_size // 2: j_val + patch_size // 2 + 1]
            samples = np.empty((K, patch_size * patch_size), dtype=np.float32)
            
            # Generate random samples
            for k in range(K):
                i_offset = np.random.randint(i_val - random_R, i_val + random_R + 1)
                j_offset = np.random.randint(j_val - random_R, j_val + random_R + 1)
                
                sample_patch = noisy_padded[i_offset: i_offset + patch_size,
                                          j_offset: j_offset + patch_size]
                samples[k] = sample_patch.reshape(-1)
            
            # Calculate max eigenvalue
            covariance_matrix = np.matmul(samples, samples.T)
            eigenvalues, _ = np.linalg.eig(covariance_matrix)
            max_eigenvalue = np.max(eigenvalues)
            
            # Store max eigenvalue
            max_eigenvalues[i_val - patch_size // 2: i_val + patch_size // 2 + 1,
                          j_val - patch_size // 2: j_val + patch_size // 2 + 1] = max_eigenvalue
    
    # Plot max eigenvalues
    plt.figure(figsize=(8, 8))
    plt.imshow(max_eigenvalues, cmap='gray')
    plt.title('Maximum Eigenvalues')
    plt.colorbar()
    plt.axis('off')
    plt.show()
    
    return max_eigenvalues

if __name__ == "__main__":
    img_path = 'barbara.tif'
    max_eigenvalues = plot_max_eigenvalues(img_path)
