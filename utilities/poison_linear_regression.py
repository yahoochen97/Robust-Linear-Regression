import numpy as np


def poison_linear_regression(X, y, feature_range, X_c, y_c, param, max_iter=2000):
    """
    Generate additional adversarial training data.
    :param X, y: the training data and lables, each feature is normalized in (0, 1)]
    :param feature_range:
    :param X_c: set of initial data vectors x_c - q by k
    :param y_c: set of initial data labels - q by 1
    :param max_iter:
    :param param:
    :return: final attack points
    """

    lam = param['lam']
    beta = param['beta']
    sigma = param['sigma']
    eps = param['eps']

    y_all = np.concatenate([y, y_c])
    w = np.zeros(X.shape[1])
    b = 0
    for iter in range(max_iter):
        obj_value = attack_objective(X_c, y_c, w, b, lam)
        for ind, _ in enumerate(X_c):
            X_all = np.concatenate([X, X_c], axis=0)
            w, b = train_classifier(X_all, y_all, lam)  # learn classifier w and b on new data X_all, y_all

            # compute objective gradient on original data, excluding the attack samples
            grad = compute_gradient(X, y, X_c[ind], y_c[ind], w, b, lam)
            update_direction = box_projection(X_c[ind] + grad, feature_range) - X_c[ind]

            # line search to set the proper gradient ascent step size eta
            for k in range(100):
                eta = beta ** k
                # update x_c along update_direction update_direction to maximize error
                obj_new = attack_objective(X_c[ind] + eta * update_direction, y_c[ind], w, b, lam)
                obj_old = attack_objective(X_c[ind], y_c[ind], w, b, lam)
                if obj_new <= obj_old - sigma * eta * np.linalg.norm(update_direction, 2):
                    X_c[ind] = X_c[ind] + eta * update_direction
                    break

        obj_diff = attack_objective(X_c, y_c, w, b, lam) - obj_value
        # print('Iter=', iter, 'objective value =', obj_value, 'improvement=', obj_diff)
        if abs(obj_diff) < eps:
            break

    return X_c


def train_classifier(X, y, lam):
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
    grad_mtrx = (X.dot(w) + b * np.ones([n, 1]) - y).T.dot(X.dot(dw) + np.ones([n, 1]).dot(db)) + lam * r.dot(dw)
    grad_vec = np.mean(grad_mtrx, axis=0)

    return grad_vec


def box_projection(x, feature_range):

    low = feature_range[0]
    upp = feature_range[1]
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

