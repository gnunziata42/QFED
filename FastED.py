import numpy as np
from scipy import ndimage

class FastQFED:
    """Fast Quantum Fuzzy Edge Detector using precomputed lookup tables for gradient values.
    This class uses a mapping table to quickly determine edge values based on gradient inputs.
    Args:
        table (str): Name of the lookup table to use. Default is "quantum". Choices are "classic", "simulator", "noisysim", and "quantum".
    """
    def __init__(self, table="quantum"):
        # Define the gradient kernels for x and y directions
        self.Gx = np.array([[-1, 1]])
        self.Gy = np.array([[-1], [1]])

        # Initialize the gradients range
        self.I_x = np.arange(-1, 1.1, 0.1)
        self.I_y = np.arange(-1, 1.1, 0.1)

        # Lookup tables files relative to the library path
        self.table_paths = {
            "classic": "mapping_classic.npy",
            "simulator": "mapping_simulator.npy",
            "noisysim": "mapping_noisysim.npy",
            "quantum": "mapping_quantum.npy",
        }

        table_path = self.table_paths.get(table.lower(), self.table_paths["quantum"])
        self.mapping = np.load(table_path)

    def detect_edges(self, grayscale):
        """Detect edges in a grayscale image using precomputed lookup table.
        Args:
            grayscale (np.ndarray): Input grayscale image.
        Returns:
            np.ndarray: Detected edges in the image."""
        # Compute gradients with convolution
        cdx = ndimage.convolve(grayscale, self.Gx, mode='constant')
        cdy = ndimage.convolve(grayscale, self.Gy, mode='constant')

        # Map gradient to index 
        x_indices = np.clip(((cdx - self.I_x[0]) / 0.1).round().astype(int), 0, len(self.I_x) - 1)
        y_indices = np.clip(((cdy - self.I_y[0]) / 0.1).round().astype(int), 0, len(self.I_y) - 1)

        # Lookup mapping
        edge = self.mapping[x_indices, y_indices]
        return edge