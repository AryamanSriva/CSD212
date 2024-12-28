import matplotlib.pyplot as plt
from utils import load_image

def show_original_image(image_path):
    # Load and display original image
    img_gray = load_image(image_path)
    
    plt.figure(figsize=(5,5))
    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return img_gray

if __name__ == "__main__":
    img_path = 'barbara.tif'
    original_img = show_original_image(img_path)
