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
    X_star, U_star = generate_pristine_data(n, k, m)

    # poisoning for subspace recovery
    X_a1 = poison_subspace_recovery(X_star, n1, k, m)
    X_all = np.concatenate([X_star, X_a1], axis=0)
    robust_recovery = RSR(X_all, n, n1, k, max_iters=100)
    assignments, U, B = robust_recovery.recover()

    # poisoning for linear regression
    w_star = np.random.rand(k)
    y_star = U_star.dot(w_star)
    ind_adv_seeds = np.random.choice(n, n1, replace=False)
    X_c = U_star[ind_adv_seeds]
    y_c = y_star[ind_adv_seeds]
    lam = 0.001
    betta = 0.5
    sigma = 0.01
    eps = 0.01

    X_a2 = poison_linear_regression(U_star, y_star, X_c, y_c, lam, betta, sigma, eps)

if __name__ == "__main__":
    main()
