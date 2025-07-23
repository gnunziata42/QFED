import numpy as np
import skfuzzy as fuzz
import math
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from tqdm import tqdm
import multiprocessing as mp
from scipy import ndimage

class QFEDStandard:
    def __init__(self, backend, n_shots, sigma=0.1):
        '''Initialize the QFED Standard Circuit.'''
        # Set the backend and the number of shots for quantum execution
        self.backend = backend
        self.n_shots = n_shots

        # Universe of Discourse definitions
        self.Ix = np.arange(-1, 1.01, 0.01)
        self.Iy = np.arange(-1, 1.01, 0.01)
        self.P_I = np.arange(0, 1.0001, 0.01)

        # Membership definitions
        self.Ix_zero = fuzz.gaussmf(self.Ix, 0, sigma)
        self.Ix_one = 1.0 - fuzz.gaussmf(self.Ix, 0, sigma)
    
        self.Iy_zero = fuzz.gaussmf(self.Iy, 0, sigma)
        self.Iy_one = 1.0 - fuzz.gaussmf(self.Iy, 0, sigma)

        self.P_I_white = fuzz.trapmf(self.P_I, [0.1, 1, 1, 1])
        self.P_I_black = fuzz.trapmf(self.P_I, [0, 0, 0, 0.7])
        
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

    def build_circuit(self, ix, iy):
        '''Build the quantum circuit for edge detection based on the input gradient values. 
        Args:
            ix (float): Gradient value in the x direction.
            iy (float): Gradient value in the y direction.
        Returns:
            QuantumCircuit: The constructed quantum circuit for edge detection.'''
        
        # Initialize the quantum circuit with 4 qubits and 2 classical bits
        # The first two qubits will hold the input gradient values, and the last two qubits will hold the output values for the edge detection
        qc = QuantumCircuit(4, 2)

        # Fuzzification of the input gradient values
        mu_Ix_zero, mu_Ix_one = round(fuzz.interp_membership(self.Ix, self.Ix_zero, ix),4), round(fuzz.interp_membership(self.Ix, self.Ix_one, ix),4)
        mu_Iy_zero, mu_Iy_one = round(fuzz.interp_membership(self.Iy, self.Iy_zero, iy),4), round(fuzz.interp_membership(self.Iy, self.Iy_one, iy),4)

        # Initialize the quantum circuit with the fuzzified values
        qc.initialize([math.sqrt(mu_Ix_zero), math.sqrt(mu_Ix_one)], 0)
        qc.initialize([math.sqrt(mu_Iy_zero), math.sqrt(mu_Iy_one)], 1)

        # Apply the fuzzy rules using controlled gates
        # Rule 1: if Ix and Iy are both zero, then output is white
        qc.x(0)
        qc.x(1)
        qc.mcx([0, 1], 2)
        qc.x(0)
        qc.x(1)

        # Rule 2: if Ix is zero and Iy is one, then output is black
        qc.x(1)
        qc.mcx([0, 1], 3)
        qc.x(1)

        # Rule 3: if Ix is one and Iy is zero, then output is black
        qc.x(0)
        qc.mcx([0, 1], 3)
        qc.x(0)

        # Rule 4: if Ix and Iy are both one, then output is black
        qc.mcx([0, 1], 3)

        # Measure the output qubits
        qc.barrier()
        qc.measure(2, 0)
        qc.measure(3, 1)

        return qc

    def execute_circuit(self, qc):
        '''Execute the quantum circuit on the specified backend and return the counts of measurement results.
        Args:
            qc (QuantumCircuit): The quantum circuit to be executed.
        Returns:
            dict: The counts of measurement results from the quantum circuit execution.'''
        
        # Transpile the quantum circuit for the specified backend
        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=3)
        transpiled_qc = pm.run(qc)

        # Execute the quantum circuit using the SamplerV2
        sampler = SamplerV2(self.backend)
        job = sampler.run([transpiled_qc], shots=self.n_shots)

        # Get the result of the job and extract the counts
        result = job.result()
        counts = list(result[0].data._data.values())[0].get_counts()
        return counts

    def counts_evaluator(self, counts):
        '''Function returning the alpha values for alpha-cutting the output fuzzy sets according to the
        probability of measuring the related basis states on the output quantum register.
        Args:
            counts (dict): The counts of measurement results from the quantum circuit execution.
        Returns:
            dict: alpha values for alpha-cutting the output fuzzy sets.'''
        
        # Initialize the output dictionary with keys '01' and '10' for white and black edges
        output = {'01': 0, '10': 0}

        # Normalize the counts and update the output dictionary
        n_shots = sum(counts.values())
        counts = {k: v / n_shots for k, v in counts.items()}
        for key, val in counts.items():
            if key in output:
                output[key] = val
        return output

    def process_pixel(self, ix, iy):
        '''Process a single pixel's gradient values and return the defuzzified output for edge detection.
        Args:
            ix (float): Gradient value in the x direction.
            iy (float): Gradient value in the y direction.
        Returns:
            float: The defuzzified output for edge detection based on the input gradient values.'''
        
        # Build the quantum circuit for the given pixel's gradient values
        qc = self.build_circuit(ix, iy)
        counts = self.execute_circuit(qc)
        evaluated = self.counts_evaluator(counts)

        # Defuzzify the output using the alpha values for white and black edges, using the LOM method
        alpha_w = evaluated.get('01', 0)
        alpha_b = evaluated.get('10', 0)

        white_activation = np.fmin(alpha_w, self.P_I_white)
        black_activation = np.fmin(alpha_b, self.P_I_black)
        aggregated = np.fmax(white_activation, black_activation)
        return fuzz.defuzz(self.P_I, aggregated, 'LOM')

    def detect_edges(self, image, parallel=True):
        """Detect edges in an image using quantum fuzzy edge detection.
        Args:
            image (np.ndarray): Input grayscale image.
            parallel (bool): Whether to use parallel processing for edge detection.
        Returns:
            np.ndarray: Edge map of the image, where each pixel's value is the defuzzified edge value."""
        
        # Initialize the edge map with zeros
        cdx, cdy = self.compute_gradients(image)
        height, width = cdx.shape
        edge_map = np.zeros((height, width))

        # Create a list of tasks for parallel processing
        tasks = [(cdx[i, j], cdy[i, j]) for i in range(height) for j in range(width)]

        if parallel:
            with mp.Pool(mp.cpu_count()) as pool:
                results = list(tqdm(pool.imap(self._parallel_process, tasks), total=len(tasks), desc='Quantum Edge Detection'))
            edge_map = np.array(results).reshape(height, width)
        else:
            results = []
            for ix, iy in tqdm(tasks, desc='Quantum Edge Detection'):
                results.append(self.process_pixel(ix, iy))
            edge_map = np.array(results).reshape(height, width)

        return edge_map

    def _parallel_process(self, args):
        ix, iy = args
        detector = QFEDStandard(backend=self.backend, n_shots=self.n_shots)
        return detector.process_pixel(ix, iy)
    
class QFEDOptimized:
    def __init__(self, backend, n_shots, sigma=0.1):
        '''Initialize the QFED Standard Circuit.'''
        # Set the backend and the number of shots for quantum execution
        self.backend = backend
        self.n_shots = n_shots

        # Universe of Discourse definitions
        self.Ix = np.arange(-1, 1.01, 0.01)
        self.Iy = np.arange(-1, 1.01, 0.01)
        self.P_I = np.arange(0, 1.0001, 0.01)

        # Membership definitions
        self.Ix_zero = fuzz.gaussmf(self.Ix, 0, sigma)
        self.Ix_one = 1.0 - fuzz.gaussmf(self.Ix, 0, sigma)
    
        self.Iy_zero = fuzz.gaussmf(self.Iy, 0, sigma)
        self.Iy_one = 1.0 - fuzz.gaussmf(self.Iy, 0, sigma)

        self.P_I_white = fuzz.trapmf(self.P_I, [0.1, 1, 1, 1])
        self.P_I_black = fuzz.trapmf(self.P_I, [0, 0, 0, 0.7])
        
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

    def build_circuit(self, ix, iy):
        '''Build the quantum circuit for edge detection based on the input gradient values. 
        Args:
            ix (float): Gradient value in the x direction.
            iy (float): Gradient value in the y direction.
        Returns:
            QuantumCircuit: The constructed quantum circuit for edge detection.'''
        
        # Initialize the quantum circuit with 3 qubits and 1 classical bit
        # The first two qubits will hold the input gradient values, and the last one will hold the output value for the edge detection
        qc = QuantumCircuit(3, 1)

        # Fuzzification of the input gradient values
        mu_Ix_zero = round(fuzz.interp_membership(self.Ix, self.Ix_zero, ix),4)
        mu_Iy_zero = round(fuzz.interp_membership(self.Iy, self.Iy_zero, iy),4)
        
        # Initialize the quantum circuit with the fuzzified values using rotation gates
        theta_Ix, theta_Iy = round(2*math.acos(math.sqrt(mu_Ix_zero)),4), round(2*math.acos(math.sqrt(mu_Iy_zero)),4)

        qc.ry(theta_Ix, 0)
        qc.ry(theta_Iy, 1)

        # Optimized rule set using a Toffoli gate
        qc.x(0)
        qc.x(1)

        qc.mcx([0,1],2)

        qc.x(0)
        qc.x(1)
        qc.x(2)
        
        qc.measure(2,0)
        return qc

    def execute_circuit(self, qc):
        '''Execute the quantum circuit on the specified backend and return the counts of measurement results.
        Args:
            qc (QuantumCircuit): The quantum circuit to be executed.
        Returns:
            dict: The counts of measurement results from the quantum circuit execution.'''
        
        # Transpile the quantum circuit for the specified backend
        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=3)
        transpiled_qc = pm.run(qc)

        # Execute the quantum circuit using the SamplerV2
        sampler = SamplerV2(self.backend)
        job = sampler.run([transpiled_qc], shots=self.n_shots)

        # Get the result of the job and extract the counts
        result = job.result()
        counts = list(result[0].data._data.values())[0].get_counts()
        return counts

    def counts_evaluator(self, counts):
        '''Function returning the alpha values for alpha-cutting the output fuzzy sets according to the
        probability of measuring the related basis states on the output quantum register.
        Args:
            counts (dict): The counts of measurement results from the quantum circuit execution.
        Returns:
            dict: alpha values for alpha-cutting the output fuzzy sets.'''
        
        # Initialize the output dictionary with keys '01' and '10' for white and black edges
        output = {'0': 0, '1': 0}

        # Normalize the counts and update the output dictionary
        n_shots = sum(counts.values())
        counts = {k: v / n_shots for k, v in counts.items()}
        for key, val in counts.items():
            if key in output:
                output[key] = val
        return output

    def process_pixel(self, ix, iy):
        '''Process a single pixel's gradient values and return the defuzzified output for edge detection.
        Args:
            ix (float): Gradient value in the x direction.
            iy (float): Gradient value in the y direction.
        Returns:
            float: The defuzzified output for edge detection based on the input gradient values.'''
        
        # Build the quantum circuit for the given pixel's gradient values
        qc = self.build_circuit(ix, iy)
        counts = self.execute_circuit(qc)
        evaluated = self.counts_evaluator(counts)

        # Defuzzify the output using the alpha values for white and black edges, using the LOM method
        alpha_w = evaluated.get('0', 0)
        alpha_b = evaluated.get('1', 0)

        white_activation = np.fmin(alpha_w, self.P_I_white)
        black_activation = np.fmin(alpha_b, self.P_I_black)
        aggregated = np.fmax(white_activation, black_activation)
        return fuzz.defuzz(self.P_I, aggregated, 'LOM')

    def detect_edges(self, image, parallel=True):
        """Detect edges in an image using quantum fuzzy edge detection.
        Args:
            image (np.ndarray): Input grayscale image.
            parallel (bool): Whether to use parallel processing for edge detection.
        Returns:
            np.ndarray: Edge map of the image, where each pixel's value is the defuzzified edge value."""
        
        # Initialize the edge map with zeros
        cdx, cdy = self.compute_gradients(image)
        height, width = cdx.shape
        edge_map = np.zeros((height, width))

        # Create a list of tasks for parallel processing
        tasks = [(cdx[i, j], cdy[i, j]) for i in range(height) for j in range(width)]

        if parallel:
            with mp.Pool(mp.cpu_count()) as pool:
                results = list(tqdm(pool.imap(self._parallel_process, tasks), total=len(tasks), desc='Quantum Edge Detection'))
            edge_map = np.array(results).reshape(height, width)
        else:
            results = []
            for ix, iy in tqdm(tasks, desc='Quantum Edge Detection'):
                results.append(self.process_pixel(ix, iy))
            edge_map = np.array(results).reshape(height, width)

        return edge_map

    def _parallel_process(self, args):
        ix, iy = args
        detector = QFEDOptimized(backend=self.backend, n_shots=self.n_shots)
        return detector.process_pixel(ix, iy)

