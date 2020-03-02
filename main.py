import numpy as np
from utilities.robust_subspace_recovery import RSR
from utilities.dgp import generate_pristine_data, poison_subspace_recovery
from utilities.poison_linear_regression import poison_linear_regression

def main():
    n = 350
    m = 50
    n1 = 50
    # for k in range(20):
    #     X_star = generate_pristine_data(n, k, m)
    #     X_A = poison_subspace_recovery(X_star, n1, k, m)

    k = 20
    X = np.random.rand(n, k)
    w = np.random.rand(k)
    y = X.dot(w)
    X_c = X[0:50, :]
    y_c = y[0:50]
    lam = 0.001
    betta = 0.5
    sigma = 0.01
    eps = 0.01
    X_a = poison_linear_regression(X, y, X_c, y_c, lam, betta, sigma, eps)

if __name__ == "__main__":
    main()
