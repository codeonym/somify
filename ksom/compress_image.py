
# For arrays manipulation
import numpy as np
# For image handling
import cv2
# Kohonen Self Organized Maps Training Model
from ksom.Ksom import SOM
from ksom.mse import mse


# Define a function to compress an image using the SOM algorithm
def compress_image(image, block_width, block_height, bits_per_codevector,
                   epochs, initial_learning_rate, grayscale=False):
    """Compress an image using the SOM algorithm and save the output."""

    # Compute the vector dimension and the codebook size
    vector_dimension = block_width * block_height * image.shape[-1]
    codebook_size = 2 ** bits_per_codevector

    # Get the image height and width
    image_height, image_width = image.shape[:2]

    # Check if grayscale option is selected
    if grayscale:
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Adjust the vector dimension
        vector_dimension = vector_dimension // 3
    else:
        # Reshape the image to have three channels in the last dimension
        image = image.reshape(image_height, image_width, 3)

    image_vectors = []

    # Divide the image into non-overlapping blocks
    for i in range(0, image_height, block_height):
        for j in range(0, image_width, block_width):
            # Flatten each block into a vector and append to the list
            block_vector = image[i:i + block_height, j:j + block_width].flatten()
            image_vectors.append(block_vector)

    image_vectors = np.asarray(image_vectors).astype(float)

    # Compute the SOM rows and columns for uncertainty ( not guaranteed perfect square root )
    som_rows = int(2 ** (int(np.log2(codebook_size)) / 2))

    som_columns = int(codebook_size / som_rows)

    # Create a SOM object with the given parameters
    som = SOM(som_rows, som_columns, vector_dimension, epochs, image_vectors.shape[0],
              initial_learning_rate, max(som_rows, som_columns) / 2)

    # Train the SOM with the image vectors
    reconstruction_values = som.train(image_vectors)

    # Find the index of the closest weight vector for each image vector
    image_vector_indices = np.argmin(np.linalg.norm(image_vectors[:, np.newaxis] - reconstruction_values,
                                                    axis=2), axis=1)

    # Create an empty array for the image after compression
    image_after_compression = np.zeros_like(image, dtype="uint8")

    # Loop over the image vectors and their indices
    for index, image_vector in enumerate(image_vectors):
        # Compute the start and end row and column for each block
        start_row = int(index / (image_width / block_width)) * block_height
        end_row = start_row + block_height
        start_column = (index * block_width) % image_width
        end_column = start_column + block_width

        # Reshape the corresponding weight vector and assign it to the block
        image_block = np.reshape(reconstruction_values[image_vector_indices[index]], (block_height, block_width, -1))

        # ...

        # Check if the image block has a single channel (grayscale)
        if image_block.shape[-1] == 1:
            # If single channel, create a 3D array with the grayscale value replicated in all three channels
            grayscale_block = np.expand_dims(image_block, axis=-1)  # Add a singleton dimension
            image_after_compression[start_row:end_row, start_column:end_column] = np.repeat(grayscale_block, 3,
                                                                                            axis=-1)[:, :, 0, 0]
        else:
            # If multiple channels, assign the RGB image block to the compressed image
            image_after_compression[start_row:end_row, start_column:end_column] = image_block

        # ...

    return image_after_compression, mse(image, image_after_compression)


