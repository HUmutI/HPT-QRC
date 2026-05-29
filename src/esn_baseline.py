import numpy as np
from sklearn.linear_model import Ridge

class EchoStateNetwork:
    def __init__(self, in_size=1, res_size=100, out_size=1, alpha=0.3, sparsity=0.2, spectral_radius=0.9, ridge_alpha=1e-4, seed=42):
        """
        Classic Echo State Network (ESN) baseline.
        """
        self.in_size = in_size
        self.res_size = res_size
        self.out_size = out_size
        self.alpha = alpha  # Leaking rate
        
        np.random.seed(seed)
        
        # Input to Reservoir weights (including bias)
        self.W_in = np.random.rand(res_size, in_size + 1) - 0.5
        
        # Reservoir to Reservoir weights
        self.W = np.random.rand(res_size, res_size) - 0.5
        
        # Apply sparsity
        mask = np.random.rand(res_size, res_size) > sparsity
        self.W[mask] = 0
        
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(self.W)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        if max_eigenvalue > 0:
            self.W = self.W * (spectral_radius / max_eigenvalue)
            
        self.readout = Ridge(alpha=ridge_alpha)
        
    def _run_reservoir(self, data, init_state=None):
        """
        data: shape (n_samples, in_size)
        returns: (design_matrix, final_state) where design_matrix has shape (n_samples, 1 + in_size + res_size)
        """
        n_samples = data.shape[0]
        if init_state is not None:
            state = init_state.copy()
        else:
            state = np.zeros((self.res_size, 1))
        
        # The design matrix collects [bias, input, reservoir_state]
        design_matrix = np.zeros((n_samples, 1 + self.in_size + self.res_size))
        
        for t in range(n_samples):
            u = data[t:t+1, :].T  # input at time t, shape (in_size, 1)
            
            # Add bias term 1 to input -> (in_size + 1, 1)
            u_with_bias = np.vstack((np.array([[1.0]]), u))
            
            # Update reservoir state
            state_update = np.tanh(np.dot(self.W_in, u_with_bias) + np.dot(self.W, state))
            state = (1 - self.alpha) * state + self.alpha * state_update
            
            # Store in design matrix
            design_matrix[t, :] = np.vstack((np.array([[1.0]]), u, state)).flatten()
            
        return design_matrix, state

    def fit(self, X, y, discard_steps=100):
        """
        X: shape (n_samples, in_size)
        y: shape (n_samples, out_size)
        discard_steps: number of initial steps to discard to let reservoir initialization wash out.
        """
        states, self.last_state = self._run_reservoir(X)

        # Discard initial washout transient
        states = states[discard_steps:, :]
        targets = y[discard_steps:, :]

        self.readout.fit(states, targets)

    def predict(self, X):
        states, self.last_state = self._run_reservoir(X, init_state=self.last_state)
        return self.readout.predict(states)
