import tensorflow as tf
import matplotlib.pyplot as plt
from part_a import img_a
from part_f import mask_image

# Add mask to original image
sharpened_image = img_a + mask_image

# Print information
print('Sharpened Image Data_type is', sharpened_image.dtype)
print('Sharpened Image Size is', sharpened_image.shape)
print('Sharpened Image min and max are ({}, {})'.format(tf.reduce_min(sharpened_image),
                                                    tf.reduce_max(sharpened_image)))

# Display result
plt.figure(figsize=(10, 10))
plt.imshow(tf.squeeze(sharpened_image), cmap='gray')
plt.title('Sharpened Image (Sum of Original and Mask)')
plt.axis('off')
plt.show()
