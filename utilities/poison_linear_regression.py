import numpy as np


def poison_linear_regression(X, y, X_c, y_c, lam, betta, sigma, eps):
    """
    Generate additional adversarial training data.
    :param X, y: the training data and lables, each feature is normalized in (0, 1)
    :param X_c: set of initial data vectors x_c - q by k
    :param y_c: set of initial data labels - q by 1
    :param lam: regularization coefficient for the linear classifier
    :param betta, sigma, eps: betta in (0, 1), sigma and eps are small positive contants
    :return: final attack points
    """
    max_iter = 100
    for iter in range(max_iter):
        X_new = np.concatenate([X, X_c], axis=0)
        y_new = np.concatenate([y, y_c])
        # learn classifier w and b on new data X_new, y_new
        w, b = learn_LRclassifier(X_new, y_new, lam)
        X_c_old = X_c
        for ind, x_c in enumerate(X_c_old):
            # compute objective gradient on original data, excluding the atack samples
            delta_W = compute_gradient(X, y, x_c, y_c[ind], w, b, lam)
            d = box_projection(x_c + delta_W, 0, 1) - x_c
            # line search to set the proper gradient step etta
            x_c_old = x_c
            for k in range(10):
                etta = pow(betta, k + 1)
                # update x_c along direction d to maximize error
                x_c = x_c_old + etta * d
                if attack_objective(x_c, y_c[ind], w, b, lam) <= (
                        attack_objective(x_c_old, y_c[ind], w, b, lam) - sigma * etta * np.linalg.norm(d, 2)):
                    break
            X_c[ind] = x_c
        if eps < attack_objective(X_c, y_c, w, b, lam) - attack_objective(X_c_old, y_c, w, b, lam) < eps:
            break

    return X_c


def learn_LRclassifier(X, y, lam):
    """
    Given data and labels, output the linear regression classifier
    :param X_tl:
    :param y:
    :param lamda:
    :return:
    """
    n, d = np.shape(X)
    X_tl = np.concatenate([X, np.ones([n, 1])], axis=1)
    w_tl = np.linalg.inv(X_tl.T.dot(X_tl) + lam * np.identity(d + 1)).dot(X_tl.T).dot(y)
    w = w_tl[0:-1]
    b = w_tl[-1]

    return w, b


def compute_gradient(X, y, x_c, y_c, w, b, lam):

    """compute gradient of w and b w.r.t. x_c"""
    n, d = np.shape(X)
    Sig = 1/n * X.T.dot(X)  # shape: d by d
    mu = 2/n * X.T.dot(np.ones([n, 1]))  # shape: d by 1
    M = x_c.dot(w.T) + (w.T.dot(x_c) + b - y_c) * np.identity(d)  # shape: d by d
    v = np.identity(d)

    # solve linear system to obtain
    A = np.concatenate([np.concatenate([Sig + lam * v, mu.reshape(d, 1)], axis=1), np.concatenate([mu.reshape(d, 1).T, np.ones([1, 1])], axis=1)], axis=0)
    B = -1/n * np.concatenate([M, w.reshape(d, 1).T], axis=0)
    solution_vec = np.linalg.solve(A, B)
    dw = solution_vec[0:d]
    db = solution_vec[-1].reshape(1, d)

    """compute gradient"""
    # compute gradient
    r = w.reshape(d, 1).T  # r is the gradient of regularization term w.r.t. w, for regression it is w^T
    grad = (X.dot(w) + b * np.ones([n, 1]) - y).T.dot(X.dot(dw) + np.ones([n, 1]).dot(db)) + lam * r.dot(dw)
    grad = np.mean(grad, axis=0)

    return grad


def box_projection(x, low, upp):

    d = len(x)
    x_projected = np.zeros(d)
    for ind in range(d):
        x_projected[ind] = min(upp, max(low, x[ind]))

    return x_projected


def attack_objective(X, y, w, b, lam):

    if np.ndim(X) == 1:
        X = X.reshape(1, len(X))
    n, d = np.shape(X)
    obj_value = 1/n * np.linalg.norm(X.dot(w) + b * np.ones(n) - y, 2) + 1/2 * lam * w.T.dot(w)

    return obj_value

