# linucb_util.py
import numpy as np

class LinUCB:
    def __init__(self, dim, alpha=0.6, lam=1.0):
        self.d = dim
        self.alpha = alpha
        self.a = lam * np.eye(dim)
        self.b = np.zeros(dim)

    # --- API ---
    def update(self, x, reward):
        x = np.asarray(x, dtype=float)
        self.a += np.outer(x, x)
        self.b += reward * x

    def theta(self):
        return np.linalg.solve(self.a, self.b)

    def ucb(self, x):
        x = np.asarray(x, dtype=float)
        inv = np.linalg.inv(self.a)
        mean = x @ inv @ self.b
        conf = self.alpha * np.sqrt(x @ inv @ x)
        return mean + conf