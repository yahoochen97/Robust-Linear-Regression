import numpy as np
from utilities.robust_subspace_recovery import RSR
from utilities.robust_regression import RobustRegression
from utilities.poison_subspace_recovery import generate_pristine_data, poison_subspace_recovery
from utilities.poison_linear_regression import poison_linear_regression
from sklearn.preprocessing import normalize


def recovery():
    n = 220
    m = 400
    n1 = 400 - n
    k = 10
    # for k in range(20):
    #     X_star = generate_pristine_data(n, k, m)
    #     X_A = poison_subspace_recovery(X_star, n1, k, m)

    X_star, U_star = generate_pristine_data(n, k, m, iters=1000)

    # poisoning for subspace recovery
    X_a1 = poison_subspace_recovery(X_star, n1, k, m)
    X_all = np.concatenate([X_star, X_a1], axis=0)
    robust_recovery = RSR(X_all, n, n1, k, max_iters=100)
    assignments, U, B = robust_recovery.recover()

    print(np.sum(assignments[-n1:]))
    diff = X_star-U[assignments.reshape(-1,)].dot(B.T)
    print(np.sqrt(np.mean(diff**2)))
    
    # TODO: Generate plots


def regression():
    # poisoning for linear regression
    n = 380
    m = k = 20
    n1 = 400 - n
    X_star, U_star = generate_pristine_data(n, k, m, iters=1000)

    w_star = np.random.rand(k)
    y_star = X_star.dot(w_star)
    ind_adv_seeds = np.random.choice(n, n1, replace=False)
    X_c = X_star[ind_adv_seeds]
    y_c = y_star[ind_adv_seeds]
    lam = 0.001
    betta = 0.5
    sigma = 0.01
    eps = 0.01

    X_a2 = poison_linear_regression(X_star, y_star, X_c, y_c, lam, betta, sigma, eps)
    X_all = np.concatenate([X_star, X_a2], axis=0)
    y_all = np.concatenate([y_star, y_c], axis=0)

    # TODO: Trimmed Regression
    robust_regression = RobustRegression(X_all, y_all, n, n1, k, max_iters=100)
    w_predict = robust_regression.trimmed_principal_component_regression(X_star)
    y_predict = X_star.dot(w_predict)
    rmse = np.sqrt(np.mean(pow(y_predict - y_star, 2)))
    print(rmse)


if __name__ == "__main__":
    # recovery()
    regression()
