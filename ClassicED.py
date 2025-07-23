import numpy as np
import skfuzzy as fuzz
from concurrent.futures import ProcessPoolExecutor
from scipy import ndimage
from tqdm import tqdm

class GeometricFuzzyED:
    """Geometric-based fuzzy edge detector using trapezoidal membership functions."""
    def __init__(self, gray_step=0.01, black_vector=[0, 0, 0.25, 0.40], white_vector=[0.39, 0.70, 1, 1]):
        # Initialize the grayscale range, the input trapezoidal membership functions and the output Gaussian membership function
        self.gray_range = np.arange(0, 1.0001, gray_step)
        self.edge_yes = fuzz.gaussmf(self.gray_range, 0.5, 0.01)
        self.black_mf = fuzz.trapmf(self.gray_range, black_vector)
        self.white_mf = fuzz.trapmf(self.gray_range, white_vector)

    def compute_memberships(self, neighborhood):
        """Compute the fuzzy memberships for a 3x3 neighborhood.
        Args:
            neighborhood (np.ndarray): 3x3 neighborhood of pixel values.
        Returns:
            list: List of fuzzy memberships."""
        return [
            (
                fuzz.interp_membership(self.gray_range, self.black_mf, value),
                fuzz.interp_membership(self.gray_range, self.white_mf, value)
            )
            for value in neighborhood.flatten()
        ]

    def apply_explicit_rules(self, memberships):
        """Apply explicitly defined fuzzy rules.
        Args:
            memberships (list): List of fuzzy memberships.
        Returns:
            np.ndarray: Aggregated fuzzy membership values."""
        b, w = zip(*memberships)
        b = np.array(b).reshape(3, 3)
        w = np.array(w).reshape(3, 3)

        # Rule 1
        rule1 = np.fmin(
            np.fmin(np.fmin(w[0,0], w[0,1]), w[0,2]),
            np.fmin(np.fmin(w[1,0], w[1,1]), w[1,2])
        )
        rule1 = np.fmin(rule1, np.fmin(np.fmin(b[2,0], b[2,1]), b[2,2]))
        rule1 = np.fmin(rule1, self.edge_yes)

        # Rule 2
        rule2 = np.fmin(
            np.fmin(np.fmin(b[0,0], b[0,1]), b[0,2]),
            np.fmin(np.fmin(w[1,0], w[1,1]), w[1,2])
        )
        rule2 = np.fmin(rule2, np.fmin(np.fmin(w[2,0], w[2,1]), w[2,2]))
        rule2 = np.fmin(rule2, self.edge_yes)

        # Rule 3
        rule3 = np.fmin(
            np.fmin(b[0,0], w[0,1]),
            np.fmin(w[0,2], b[1,0])
        )
        rule3 = np.fmin(rule3, np.fmin(w[1,1], w[1,2]))
        rule3 = np.fmin(rule3, np.fmin(b[2,0], w[2,1]))
        rule3 = np.fmin(rule3, w[2,2])
        rule3 = np.fmin(rule3, self.edge_yes)

        # Rule 4
        rule4 = np.fmin(
            np.fmin(w[0,0], w[0,1]),
            b[0,2]
        )
        rule4 = np.fmin(rule4, w[1,0])
        rule4 = np.fmin(rule4, w[1,1])
        rule4 = np.fmin(rule4, b[1,2])
        rule4 = np.fmin(rule4, w[2,0])
        rule4 = np.fmin(rule4, w[2,1])
        rule4 = np.fmin(rule4, b[2,2])
        rule4 = np.fmin(rule4, self.edge_yes)

        # Rule 5
        rule5 = np.fmin(
            np.fmin(b[0,0], b[0,1]),
            w[0,2]
        )
        rule5 = np.fmin(rule5, b[1,0])
        rule5 = np.fmin(rule5, w[1,1])
        rule5 = np.fmin(rule5, w[1,2])
        rule5 = np.fmin(rule5, b[2,0])
        rule5 = np.fmin(rule5, w[2,1])
        rule5 = np.fmin(rule5, w[2,2])
        rule5 = np.fmin(rule5, self.edge_yes)

        # Rule 6
        rule6 = np.fmin(
            np.fmin(w[0,0], w[0,1]),
            b[0,2]
        )
        rule6 = np.fmin(rule6, w[1,0])
        rule6 = np.fmin(rule6, w[1,1])
        rule6 = np.fmin(rule6, b[1,2])
        rule6 = np.fmin(rule6, w[2,0])
        rule6 = np.fmin(rule6, b[2,1])
        rule6 = np.fmin(rule6, b[2,2])
        rule6 = np.fmin(rule6, self.edge_yes)

        # Rule 7
        rule7 = np.fmin(
            np.fmin(b[0,0], w[0,1]),
            w[0,2]
        )
        rule7 = np.fmin(rule7, b[1,0])
        rule7 = np.fmin(rule7, w[1,1])
        rule7 = np.fmin(rule7, w[1,2])
        rule7 = np.fmin(rule7, b[2,0])
        rule7 = np.fmin(rule7, b[2,1])
        rule7 = np.fmin(rule7, w[2,2])
        rule7 = np.fmin(rule7, self.edge_yes)

        # Rule 8
        rule8 = np.fmin(
            np.fmin(w[0,0], b[0,1]),
            b[0,2]
        )
        rule8 = np.fmin(rule8, w[1,0])
        rule8 = np.fmin(rule8, w[1,1])
        rule8 = np.fmin(rule8, b[1,2])
        rule8 = np.fmin(rule8, w[2,0])
        rule8 = np.fmin(rule8, w[2,1])
        rule8 = np.fmin(rule8, b[2,2])
        rule8 = np.fmin(rule8, self.edge_yes)

        # Aggregate
        aggregated = np.fmax(
            np.fmax(np.fmax(rule1, rule2), np.fmax(rule3, rule4)),
            np.fmax(np.fmax(rule5, rule6), np.fmax(rule7, rule8))
        )

        return aggregated

    def detect_pixel(self, image, i, j):
        """Detect edge value for a pixel at (i, j) using its 3x3 neighborhood.
        Args:
            image (np.ndarray): Input grayscale image.
            i (int): Row index of the pixel.
            j (int): Column index of the pixel.
        Returns:
            float: Edge value for the pixel, defuzzified using the 'lom' method."""
        
        neighborhood = image[i-1:i+2, j-1:j+2]
        memberships = self.compute_memberships(neighborhood)
        aggregated = self.apply_explicit_rules(memberships)
        return fuzz.defuzz(self.gray_range, aggregated, 'lom')

    def detect_edges(self, image):
        """Detect edges in the image using geometric fuzzy edge detection.
        Args:
            image (np.ndarray): Input grayscale image.
        Returns:
            np.ndarray: Detected edges in the image."""
        height, width = image.shape
        edge_map = np.zeros((height - 2, width - 2))

        for i in tqdm(range(1, height - 1), desc="Geometric Edge Detection (Rows)"):
            for j in range(1, width - 1):
                edge_map[i-1, j-1] = self.detect_pixel(image, i, j)

        return edge_map

class GradientFuzzyED:
    """Gradient-based fuzzy edge detector using Gaussian membership functions."""
    def __init__(self, gray_step=0.01, gradient_step=0.01):
        # Initialize the grayscale range and the output trapezoidal membership functions
        self.x_b = np.arange(0, 1.0001, gray_step)
        self.white = fuzz.trapmf(self.x_b, [0.1, 1, 1, 1])
        self.black = fuzz.trapmf(self.x_b, [0, 0, 0, 0.7])

        # Initialize the Gaussian input membership functions for gradients, ranging from -1 to 1
        start, stop = -1, 1 + 0.001
        self.I_x = np.arange(start, stop, gradient_step)
        self.I_y = np.arange(start, stop, gradient_step)
        self.Mx = fuzz.gaussmf(self.I_x, 0, 0.1)
        self.My = fuzz.gaussmf(self.I_y, 0, 0.1)
        
        # Define the gradient kernels for convolution
        self.Gx = np.array([[-1, 1]])
        self.Gy = np.array([[-1], [1]])

    def compute_gradients(self, image):
        """Compute the gradients of the image using convolution.
        Args:
            image (np.ndarray): Input grayscale image.
        Returns:
            cdx, cdy (np.ndarray): Gradients in x and y directions.
        """
        cdx = ndimage.convolve(image, self.Gx, mode='constant')
        cdy = ndimage.convolve(image, self.Gy, mode='constant')
        return cdx, cdy

    def _process_row(self, args):
        """Process a single row of the image to compute edge values.
        Args:
            args (tuple): Contains the row index and the corresponding gradient values.
        Returns:
            i (int): Row index.
            line (list): List of edge values for the row."""
        
        i, cdx_row, cdy_row = args
        # Fuzzification of the gradient values
        Ievalx = fuzz.interp_membership(self.I_x, self.Mx, cdx_row)
        Ievaly = fuzz.interp_membership(self.I_y, self.My, cdy_row)
        line = []

        for j in range(len(cdx_row)):
            # Apply the fuzzy rules to compute the edge value
            # Rule 1: If gradient in x and y are both zero, then the pixel is not an edge (white)
            # Rule 2: If one or either gradient in x and y are one, then the pixel is an edge (black)
            rule1 = np.fmin(Ievalx[j], Ievaly[j])
            rule2 = np.fmax(1 - Ievalx[j], 1 - Ievaly[j])
            pixel_white = np.fmin(rule1, self.white)
            pixel_black = np.fmin(rule2, self.black)
            # Aggregate the results using maximum
            aggregated = np.fmax(pixel_white, pixel_black)
            # Defuzzification to get the final edge value using the 'lom' method for defuzzification
            edge_value = fuzz.defuzz(self.x_b, aggregated, 'lom')
            line.append(edge_value)

        return i, line

    def detect_edges(self, image, parallel=True):
        """Detect edges in the image using gradient fuzzy edge detection.
        Args:
            image (np.ndarray): Input grayscale image.
            parallel (bool): Whether to use parallel processing.
        Returns:
            edge_map (np.ndarray): Detected edges in the image.
        """
        cdx, cdy = self.compute_gradients(image)
        height = image.shape[0]

        args_list = [
            (i, cdx[i], cdy[i])
            for i in range(height)
        ]

        edge_map = [None] * height

        if parallel:
            with ProcessPoolExecutor() as executor:
                for i, row in tqdm(executor.map(self._process_row, args_list), total=height, desc='Gradient Fuzzy Edge Detection (Rows)'):
                    edge_map[i] = row
        else:
            for args in tqdm(args_list, desc='Gradient Fuzzy Edge Detection (Rows)'):
                i, row = self._process_row(args)
                edge_map[i] = row

        return np.array(edge_map)