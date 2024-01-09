
# For arrays manipulation
import numpy as np
# For nearest neighbor calculation
from scipy.spatial import cKDTree


# Class for the self-organizing map (SOM) algorithm
class SOM:
    def __init__(self, rows, columns, dimensions, epochs, number_of_input_vectors, alpha, sigma):
        """Initialize the SOM with the given parameters."""
        self.rows = rows
        self.columns = columns
        self.dimensions = dimensions
        self.epochs = epochs
        self.alpha = alpha
        self.sigma = sigma
        self.number_of_input_vectors = number_of_input_vectors
        self.number_of_iterations = self.epochs * self.number_of_input_vectors

        # Initialize the weight vectors randomly
        self.weight_vectors = np.random.uniform(0, 255, (self.rows * self.columns, self.dimensions))

    def get_bmu_location(self, input_vector):
        """Find the best matching unit (BMU) for the given input vector."""

        # Create a cKDTree from the weight vectors
        tree = cKDTree(self.weight_vectors)

        # Query the tree for the nearest neighbor of the input vector
        bmu_index = tree.query(input_vector)[1]

        # Convert the index to a 2D location
        return np.array([int(bmu_index / self.columns), bmu_index % self.columns])

    def update_weights(self, iter_no, bmu_location, input_data):
        """Update the weight vectors according to the learning rules."""

        # Compute the learning rate and the neighborhood radius
        learning_rate_op = 1 - (iter_no / float(self.number_of_iterations))
        alpha_op = self.alpha * learning_rate_op
        sigma_op = self.sigma * learning_rate_op

        # Create an array of 2D coordinates for each neuron in the SOM grid
        neuron_coordinates = np.array([[i // self.columns, i % self.columns] for i in range(self.rows * self.columns)])

        # Calculate vectors from the BMU to each neuron
        vectors_from_bmu = neuron_coordinates - bmu_location

        # Calculate the Euclidean distance for each vector
        distance_from_bmu = np.linalg.norm(vectors_from_bmu, axis=1)

        # Compute the neighborhood function
        neighbourhood_function = np.exp(-0.5 * (distance_from_bmu ** 2) / (sigma_op ** 2))

        # Compute the final learning rate
        final_learning_rate = alpha_op * neighbourhood_function

        # Calculate the element-wise difference between input_data and weight_vectors
        difference_matrix = input_data - self.weight_vectors

        # Adjust the learning rate for each neuron
        adjusted_learning_rates = final_learning_rate[:, np.newaxis]

        # Perform element-wise multiplication
        weight_delta = adjusted_learning_rates * difference_matrix

        # Update the weight vectors
        self.weight_vectors += weight_delta

    def train(self, input_data):
        """Train the SOM with the given input data."""

        # Initialize the iteration counter
        iter_no = 0

        # Loop over the epochs
        for _ in range(self.epochs):

            # Loop over the input vectors
            for input_vector in input_data:

                # Find the BMU location
                bmu_location = self.get_bmu_location(input_vector)

                # Update the weights
                self.update_weights(iter_no, bmu_location, input_vector)

                # Increment the iteration counter
                iter_no += 1

        # Return the trained weight vectors
        return self.weight_vectors
