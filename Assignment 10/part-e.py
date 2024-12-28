from utils import RMSD, load_image
from part_d import denoise_image

def calculate_rmse(image_path):
    # Load original image
    original_img = load_image(image_path)
    
    # Get denoised image
    denoised_img = denoise_image(image_path)
    
    # Calculate RMSE
    rmse_value = RMSD(original_img, denoised_img)
    print(f"RMSE between original and denoised images: {rmse_value:.4f}")
    
    return rmse_value

if __name__ == "__main__":
    img_path = 'barbara.tif'
    rmse = calculate_rmse(img_path)
