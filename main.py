import numpy as np
from utilities.robust_subspace_recovery import RSR

def main():
    n = 1000
    n1 = 100
    m = 10
    k = 5
    epsilon = 0.1
    std = 0.1

    # generate low rank x
    low_rank_x = np.zeros((n, k))
    for i in range(k):
        low_rank_x[:,i] = np.random.normal(0, 1, size=(n,))

    # project low rank x to higher dimensional space
    projection = 2*np.random.rand(k, m) - 1
    pristine_x = low_rank_x.dot(projection)

    # generate y
    weights = 2*np.random.rand(m,1) - 1
    pristine_y = pristine_x.dot(weights)

    # poison pristine data
    poisoned_x = pristine_x + 2*epsilon*np.random.rand(n, m) - epsilon
    poisoned_y = pristine_y + np.random.normal(0, std, size=(n,1))

    # generate poison data
    insert_x = 2*np.random.rand(n1, m) - 1
    insert_y = np.random.normal(0, 1, size=(n1,1))

    # define dataset
    X = np.vstack((poisoned_x, insert_x))
    y = np.vstack((poisoned_y, insert_y))

    # robust subspace recovery
    rsr = RSR(X, y, n, n1, k, max_iters=10)

    assignments, U, B = rsr.recover()

    print(np.sum(assignments[:n]))
    print(np.sum(assignments[n:]))
    print(projection-B.T)


if __name__ == "__main__":
    main()
