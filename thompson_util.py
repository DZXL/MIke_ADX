# thompson_util.py

import numpy as np

class ThompsonSampling:
    def __init__(self, dim, lam=1.0, v=2.0):
        self.d = dim
        self.v = v
        self.A = lam * np.eye(dim)
        self.b = np.zeros(dim)

    def update(self, x, reward):
        x = np.asarray(x, dtype=float)
        self.A += np.outer(x, x)
        self.b += reward * x

    def sample_theta(self):
        cov = self.v ** 2 * np.linalg.inv(self.A)
        mean = np.linalg.solve(self.A, self.b)
        return np.random.multivariate_normal(mean, cov)

    def ucb(self, x):
        theta = self.sample_theta()
        return float(np.dot(theta, x))