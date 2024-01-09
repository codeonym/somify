
# For arrays manipulation
import numpy as np


# Define a function to compute the mean squared error (MSE) between two images
def mse(image_a, image_b):
    """Return the MSE between two images."""
    # Flatten the 3D arrays to 1D arrays
    flattened_a = image_a.flatten()
    flattened_b = image_b.flatten()

    # Calculate mean squared error
    mse_value = np.mean((flattened_a - flattened_b) ** 2)

    return mse_value

