import numpy as np


def poison_linear_regression(X, y, X_c, y_c, lam, betta, sigma, eps):
    '''
    Generate additional adversarial training data.
    :param X, y: the training data and lables
    :param X_c: set of initial data vectors x_c - q by k
    :param y_c: set of initial data labels - q by 1
    :return: final attack points
    '''
    max_iter = 100
    for iter in range(max_iter):
        X_new = np.concatenate([X, X_c], axis=0)
        y_new = np.concatenate([y, y_c])
        w, b = learn_LRclassifier(X_new, y_new, lam)
        X_c_old = X_c
        for ind, x_c in enumerate(X_c):
            # learn classifier w and b on new data X_new, y_new
            delta_W = compute_subgradient(X_new, y_new, x_c, w, b)
            d = box_projection(x_c + delta_W, 0, 1) - x_c
            # line search to set the gradient step etta
            for k in range(10):
                etta = betta ^ k
                x_c_old = x_c
                # update x_c along direction d to maximize error
                x_c = x_c + etta * d
                X_c[ind] = x_c
                if attack_objective(x_c, y_c[ind], w) <= (
                        attack_objective(x_c_old, y_c[ind], w) - sigma * etta * np.linalg.norm(d, 2)):
                    break
        if eps < attack_objective(X_c, y_c, w) - attack_objective(X_c_old, y_c, w) < eps:
            break

    return X_c


def learn_LRclassifier(X, y, lam):
    '''
    Given data and labels, output the linear regression classifier
    :param X:
    :param y:
    :param lamda:
    :return:
    '''

    return w, b


def compute_subgradient(X, y, x_c, w, b):


    return subgradient


def box_projection(x, lowerbound, upperbound):

    return x_projected


def attack_objective(X, y, w):

    return objective_value