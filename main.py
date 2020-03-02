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

    X_star = generate_pristine_data(n, k, m)
    X_a1 = poison_subspace_recovery(X_star, n1, k, m)


    w_star = np.random.rand(m)
    y_star = X_star.dot(w_star)

    robust_recovery = RSR(X_star, y_star, n, n1, k, max_iters=100)
    assignments, U, B = robust_recovery.recover()

    ind_adv_seeds = np.random.choice(n, n1, replace=False)
    X_c = X_star[ind_adv_seeds]
    y_c = y_star[ind_adv_seeds]
    lam = 0.001
    betta = 0.5
    sigma = 0.01
    eps = 0.01
    X_a2 = poison_linear_regression(X_star, y_star, X_c, y_c, lam, betta, sigma, eps)

if __name__ == "__main__":
    main()
