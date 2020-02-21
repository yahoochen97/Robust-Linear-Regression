import numpy as np


class RSR:
    def __init__(self, X, y, n, n1, k, max_iters=100):
        '''
        Trimmed optimization for subspace recovery

        Inputs:
        - X: training data
        - y: training label
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
        assignments = np.zeros((self.n1+self.n,1)).astype(bool)
        idx = np.random.choice(self.n+self.n1, self.n, replace=False)
        assignments[idx] = 1

        it = 1
        while it <= self.max_iters:
            weights = self.optimize_parameters(assignments, x, y)
            losses = [0 for _ in range(self.n1+self.n)]
            for i in range(self.n1+self.n):
                losses[i] = np.linalg.norm(y[i]-weights.dot(x[i]))

            new_idx = np.argpartition(losses, self.n)[:self.n]
            new_assignments = np.zeros((self.n1+self.n,1)).astype(bool)
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
        weights= np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y)).T
        return weights


    def recover(self):
        '''
        Efficient Robust Subspace Recovery.

        Inputs:
        - max_iters: maximal iterations for optimization

        Outputs:
        - U: numpy array of shape n by k
        - B: numpy array of shape m by k
        '''

        n, m = self.X.shape
        U = np.random.rand(n, self.k)
        B = np.random.rand(m, self.k)

        it = 1
        while it <= self.max_iters:
            U = self.optimize_U(B)
            # TODO: optimize for B
            B, assignments = self.optimize_B(U)
            it += 1

        return assignments, U, B

    def optimize_U(self, B):
        '''
        Optimize for U.

        Inputs:
        - B: Low ranked basis matrix, numpy array of shape m by k

        Outputs:
        - new_U: alternative optimized U
        '''

        # Closed form for optimal U: XB = UB^TB
        new_U = self.X.dot(B).dot(np.linalg.inv(B.T.dot(B)))
        return new_U

    def optimize_B(self, U):
        '''
        Optimize for B.

        Inputs:
        - U: Projection of X into low-ranked subspace, numpy array of shape n by k

        Outputs:
        - new_B: alternative optimized B
        '''
        new_B, new_assignments = self.trimmed_optimization(U, self.X)
        return new_B, new_assignments
