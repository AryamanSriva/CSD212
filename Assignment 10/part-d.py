import numpy as np
import matplotlib.pyplot as plt
from utils import make_noisy, load_image

def denoise_image(image_path):
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
    
    # Process each patch
    for i_val in range(neighbour_size, noisy_padded.shape[0] - neighbour_size, patch_size):
        for j_val in range(neighbour_size, noisy_padded.shape[1] - neighbour_size, patch_size):
            # Extract current patch
            pij = noisy_padded[i_val - patch_size // 2: i_val + patch_size // 2 + 1,
                             j_val - patch_size // 2: j_val + patch_size // 2 + 1]
            p_bar = pij.reshape(-1)
            
            # Collect random patches
            xij_t = np.empty((K, patch_size * patch_size), dtype=np.float32)
            for x in range(K):
                i_offset = np.random.randint(i_val - random_R, i_val + random_R + 1)
                j_offset = np.random.randint(j_val - random_R, j_val + random_R + 1)
                
                p_sample = noisy_padded[i_offset: i_offset + patch_size,
                                      j_offset: j_offset + patch_size]
                xij_t[x] = p_sample.reshape(-1)
            
            # Calculate covariance and eigenvectors
            xij = np.transpose(xij_t)
            cij = np.matmul(xij, xij_t)
            wij, vij = np.linalg.eig(cij)
            vij_t = np.transpose(vij)
            
            # Calculate coefficients
            aij = np.matmul(vij_t, p_bar)
            aij_neg2 = np.zeros(patch_size * patch_size, dtype=np.float32)
            sigma = 0.1 * (img_gray.max() - img_gray.min())
            
            for k in range(1, K):
                aij_neg2 += np.square(np.matmul(vij_t, xij[:, k])) - np.square(sigma)
            
            aij_neg2 = np.maximum(0, aij_neg2 / K)
            bij = aij / (1 + (np.square(sigma) / aij_neg2))
            
            # Reconstruct denoised patch
            qij_bar = np.matmul(vij, bij)
            qij = qij_bar.reshape(patch_size, patch_size)
            noisy_padded[i_val - patch_size // 2: i_val + patch_size // 2 + 1,
                        j_val - patch_size // 2: j_val + patch_size // 2 + 1] = qij
    
    # Remove padding to get final denoised image
    denoised_img = noisy_padded[neighbour_size:-neighbour_size,
                               neighbour_size:-neighbour_size]
    
    # Display results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(noisy_img, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(denoised_img, cmap='gray')
    plt.title('Denoised Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return denoised_img

if __name__ == "__main__":
    img_path = 'barbara.tif'
    denoised_img = denoise_image(img_path)
