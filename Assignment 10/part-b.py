import matplotlib.pyplot as plt
from utils import make_noisy, load_image

def show_corrupted_image(image_path):
    # Load original image
    img_gray = load_image(image_path)
    
    # Add Gaussian noise
    noisy_img = make_noisy("gauss", img_gray)
    
    # Display corrupted image
    plt.figure(figsize=(5,5))
    plt.imshow(noisy_img, cmap='gray')
    plt.axis('off')
    plt.title('Corrupted Image')
    plt.show()
    
    return noisy_img

if __name__ == "__main__":
    img_path = 'barbara.tif'
    noisy_img = show_corrupted_image(img_path)
