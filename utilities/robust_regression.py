import numpy as np
import scipy
from utilities.robust_subspace_recovery import RSR


class RobustRegression:
    def __init__(self, X, y, n, n1, k, max_iters=100):
        '''
        Trimmed optimization for subspace recovery

        Inputs:
        - X: training data
        - n: number of pristine examples
        - n1: number of poisoning examples
        - k: rank of subspace
        - weights: initial weights
        - max_iters: maximal iterations for optimization
        '''
        self.X = X
        self.y = y
        self.n = n
        self.n1 = n1
        self.k = k
        self.max_iters = max_iters

    def trimmed_optimization(self, x, y):
        '''
        Trimmed optimization for subspace recovery

        Inputs:
        - x: training samples for optimization
        - y: training labels
        - max_iters: maximal iterations for optimization
        Outputs:
        - weights: optimized parameters
        - new_assignments: binary numpy array (n+n1) by 1, should sum to n
        '''

        # initialize random assignment
        assignments = np.zeros((self.n1 + self.n, 1)).astype(bool)
        idx = np.random.choice(self.n + self.n1, self.n, replace=False)
        assignments[idx] = 1

        it = 1
        while it <= self.max_iters:
            weights = self.optimize_parameters(assignments, x, y)
            losses = [0 for _ in range(self.n1 + self.n)]
            for i in range(self.n1 + self.n):
                losses[i] = pow(y[i] - weights.dot(x[i]), 2)

            new_idx = np.argpartition(losses, self.n)[:self.n]
            new_assignments = np.zeros((self.n1 + self.n, 1)).astype(bool)
            new_assignments[new_idx] = 1

            if np.equal(assignments, new_assignments).all():
                break
            else:
                assignments = new_assignments
            it += 1

        return weights, new_assignments

    def optimize_parameters(self, assignments, x, y):
        '''
        Optimization for parameter estimation

        Inputs:
        - assignments: binary numpy array (n+n1) by 1, should sum to n
        - x: training examples for optimiztion
        - y: training labels
        Outputs:
        - weights: optimized parameters
        '''
        x = x[assignments.reshape(-1,)]
        y = y[assignments.reshape(-1,)]
        weights = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y)).T
        return weights

    def trimmed_principal_component_regression(self):
        """

        :param B: a basis from X using Algorithm 2 in paper
        :return: weight
        """
        robust_recovery = RSR(self.X, self.n, self.n1, self.k, self.max_iters)
        assignments, U, B = robust_recovery.recover()

        u, s, vh = np.linalg.svd(B)
        B = u.dot(vh)
        U = self.X.dot(B)
        weight_U, assignment = self.trimmed_optimization(U, self.y)
        weight = B.dot(weight_U)

        return weight

