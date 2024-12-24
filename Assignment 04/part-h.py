import tensorflow as tf
import matplotlib.pyplot as plt
from part_g import sharpened_image

# Apply power-law transformation
gamma = 0.6
final_result = tf.pow(sharpened_image, gamma)

# Print information
print('Final Result Data_type is', final_result.dtype)
print('Final Result Size is', final_result.shape)
print('Final Result min and max are ({}, {})'.format(tf.reduce_min(final_result),
                                                 tf.reduce_max(final_result)))

# Display result
plt.figure(figsize=(10, 10))
plt.imshow(tf.squeeze(final_result), cmap='gray')
plt.title('Final Result (Power-law Transformation)')
plt.axis('off')
plt.show()
