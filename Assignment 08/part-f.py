import numpy as np
from matplotlib import pyplot as plt

def create_meshgrid(img_shape):
    """
    Create meshgrid for coordinate locations
    
    Parameters:
    img_shape (tuple): Shape of the image (height, width)
    
    Returns:
    tuple: Center coordinates, meshgrid arrays, distance array
    """
    print(img_shape)
    c_x = np.floor(img_shape[0]/2)
    c_y = np.floor(img_shape[1]/2)
    print((c_x, c_y))
    
    [ii, jj] = np.meshgrid(range(img_shape[0]), 
                          range(img_shape[1]), 
                          indexing='ij')
    
    dist = np.sqrt(np.power(ii-c_x, 2) + np.power(jj-c_y, 2))
    
    plt.figure(figsize=(3,3))
    plt.imshow(dist, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return (c_x, c_y), (ii, jj), dist

if __name__ == "__main__":
    from part_a import load_and_display_image
    img_path = 'ckt-board-orig.tif'
    img_a0, _ = load_and_display_image(img_path)
    centers, meshgrid, dist = create_meshgrid(img_a0.shape)
