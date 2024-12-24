import tensorflow as tf
import matplotlib.pyplot as plt
from part_c import sharpened_img
from part_e import smoothed_sobel_gradient

# Create mask image
mask_image = sharpened_img * smoothed_sobel_gradient

# Print information
print('Mask Image Data_type is', mask_image.dtype)
print('Mask Image Size is', mask_image.shape)
print('Mask Image min and max are ({}, {})'.format(tf.reduce_min(mask_image),
                                               tf.reduce_max(mask_image)))

# Display result
plt.figure(figsize=(10, 10))
plt.imshow(tf.squeeze(mask_image), cmap='gray')
plt.title('Mask Image')
plt.axis('off')
plt.show()
