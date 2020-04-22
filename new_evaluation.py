import numpy as np
import time
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from utilities.robust_subspace_recovery import RSR
from utilities.robust_regression import RobustRegression
from utilities.poison_subspace_recovery import generate_pristine_data, poison_subspace_recovery
from utilities.new_poison import new_poisoning, train_classifier
from sklearn.preprocessing import normalize, MinMaxScaler


def inverse_flipping(y):

    mean = np.mean(y)
    y_out = mean * 2 - y

    return y_out


def boundary_flipping(y):

    min = np.min(y)
    max = np.max(y)
    mean = np.mean(y)
    y_out = np.zeros(len(y))

    for i, y_i in enumerate(y):
        y_out[i] = min if y_i >= mean else max

    return y_out


def label_flipping(type):

    runs = 10
    m = 20
    k = 20
    n1s = [10 + 10 * i for i in range(20)]
    RMSEs_rlr = np.zeros([len(n1s), runs])
    RMSEs_lr = np.zeros([len(n1s), runs])
    RMSEs_lro = np.zeros([len(n1s), runs])

    for idx, n1 in enumerate(n1s):

        n = 400 - n1  # number of pristine data

        for run in range(runs):
            # generate training dataset
            X_star, _ = generate_pristine_data(n, k, m, iters=1000)
            scalar = MinMaxScaler()
            scalar.fit(X_star)
            feature_range = scalar.feature_range
            X_star = scalar.transform(X_star)
            X_noised = X_star + np.random.normal(loc=0, scale=0.01, size=X_star.shape)
            w_star = np.random.rand(m)
            y_star = X_star.dot(w_star)

            # randomly generate poisoning seeds
            ind_adv_seeds = np.random.choice(n, n1, replace=False)
            X_c = X_star[ind_adv_seeds]
            y_c = y_star[ind_adv_seeds]
            X_clean = np.concatenate([X_noised, X_c], axis=0)
            y_clean = np.concatenate([y_star, y_c], axis=0)

            # poison training data
            param = dict()
            param['lam'] = 0.001
            param['beta'] = 0.5
            param['sigma'] = 1e-4
            param['eps'] = 1e-6
            X_a2 = X_c
            X_all = np.concatenate([X_noised, X_a2], axis=0)
            if type == 'inverse':
                y_all = np.concatenate([y_star, inverse_flipping(y_c)], axis=0)
            elif type == 'boundary':
                y_all = np.concatenate([y_star, boundary_flipping(y_c)], axis=0)
            else:
                print(msg)

            # train robust regression model
            robust_regression = RobustRegression(X_all, y_all, n, n1, k, max_iters=100)
            w_predict = robust_regression.trimmed_principal_component_regression()

            # train normal LR classifier
            w_lr = solve_LR(X_all, y_all)
            w_lro = solve_LR(X_clean, y_clean)

            # generate test dataset
            X_test, _ = generate_pristine_data(n, k, m, iters=1000)
            X_test = scalar.transform(X_test)
            X_test = X_test + np.random.normal(loc=0, scale=0.01, size=X_star.shape)
            y_test = X_test.dot(w_star)

            # test robust regression model
            y_predict = X_test.dot(w_predict)
            RMSE_rlr = np.sqrt(np.mean((y_predict - y_test) ** 2))
            RMSEs_rlr[idx, run] = RMSE_rlr

            # test normal LR model
            y_lr = X_test.dot(w_lr)
            RMSE_lr = np.sqrt(np.mean((y_lr - y_test) ** 2))
            RMSEs_lr[idx, run] = RMSE_lr
            y_lro = X_test.dot(w_lro)
            RMSE_lro = np.sqrt(np.mean((y_lro - y_test) ** 2))
            RMSEs_lro[idx, run] = RMSE_lro

        print('# corrupted rows', n1,
              'RMSE_rlr:', np.mean(RMSEs_rlr[idx, :]),
              'RMSE_lr:', np.mean(RMSEs_lr[idx, :]),
              'RMSE_lro', np.mean(RMSEs_lro[idx, :]))

    plt.figure()
    plt.plot(n1s, np.mean(RMSEs_rlr, axis=1), linestyle='dashed', marker='o', label='TPCR')
    plt.plot(n1s, np.mean(RMSEs_lr, axis=1), linestyle='dashed', marker='o', label='LR (O+A)')
    plt.plot(n1s, np.mean(RMSEs_lro, axis=1), linestyle='dashed', marker='o', label='LR (O)')
    plt.title('Regression Output RMSE vs # manipulated labels')
    plt.xlabel('# manipulated labels')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('result/flip_mean.png')
    plt.show()

    np.savetxt('result/flip_rlr.out', RMSEs_rlr)
    np.savetxt('result/flip_lr.out', RMSEs_lr)
    np.savetxt('result/flip_lro.out', RMSEs_lro)


def regression_new_poisoning():

    runs = 10
    m = 20
    k = 20
    n1s = [10 + 10 * i for i in range(20)]
    RMSEs_rlr = np.zeros([len(n1s), runs])
    RMSEs_lr = np.zeros([len(n1s), runs])
    RMSEs_lro = np.zeros([len(n1s), runs])

    for idx, n1 in enumerate(n1s):

        n = 400 - n1  # number of pristine data

        for run in range(runs):
            # generate training dataset
            X_star, _ = generate_pristine_data(n, k, m, iters=1000)
            scalar = MinMaxScaler()
            scalar.fit(X_star)
            feature_range = scalar.feature_range
            X_star = scalar.transform(X_star)
            X_noised = X_star + np.random.normal(loc=0, scale=0.01, size=X_star.shape)
            w_star = np.random.rand(m)
            y_star = X_star.dot(w_star)

            # randomly generate poisoning seeds
            ind_adv_seeds = np.random.choice(n, n1, replace=False)
            X_c = X_star[ind_adv_seeds]
            y_c = y_star[ind_adv_seeds]
            X_clean = np.concatenate([X_noised, X_c], axis=0)
            y_clean = np.concatenate([y_star, y_c], axis=0)

            # poison training data
            param = dict()
            param['lam'] = 0.001
            param['beta'] = 0.5
            param['sigma'] = 1e-4
            param['eps'] = 1e-6
            X_a2, y_a2 = new_poisoning(X_star, y_star, feature_range, X_c, y_c, param, max_iter=2000)
            X_all = np.concatenate([X_noised, X_a2], axis=0)
            y_all = np.concatenate([y_star, y_a2], axis=0)

            # train robust regression model
            robust_regression = RobustRegression(X_all, y_all, n, n1, k, max_iters=100)
            w_predict = robust_regression.trimmed_principal_component_regression()

            # train normal LR classifier
            w_lr = solve_LR(X_all, y_all)
            w_lro = solve_LR(X_clean, y_clean)

            # generate test dataset
            X_test, _ = generate_pristine_data(n, k, m, iters=1000)
            X_test = scalar.transform(X_test)
            X_test = X_test + np.random.normal(loc=0, scale=0.01, size=X_star.shape)
            y_test = X_test.dot(w_star)

            # test robust regression model
            y_predict = X_test.dot(w_predict)
            RMSE_rlr = np.sqrt(np.mean((y_predict - y_test) ** 2))
            RMSEs_rlr[idx, run] = RMSE_rlr

            # test normal LR model
            y_lr = X_test.dot(w_lr)
            RMSE_lr = np.sqrt(np.mean((y_lr - y_test) ** 2))
            RMSEs_lr[idx, run] = RMSE_lr
            y_lro = X_test.dot(w_lro)
            RMSE_lro = np.sqrt(np.mean((y_lro - y_test) ** 2))
            RMSEs_lro[idx, run] = RMSE_lro

        print('# corrupted rows', n1,
              'RMSE_rlr:', np.mean(RMSEs_rlr[idx, :]),
              'RMSE_lr:', np.mean(RMSEs_lr[idx, :]),
              'RMSE_lro', np.mean(RMSEs_lro[idx, :]))

    # plot RMSE_rlr
    plt.figure()
    plt.plot(n1s, np.mean(RMSEs_rlr, axis=1), linestyle='dashed', marker='o', label='TPCR')
    plt.plot(n1s, np.mean(RMSEs_lr, axis=1), linestyle='dashed', marker='o', label='LR (O+A)')
    plt.plot(n1s, np.mean(RMSEs_lro, axis=1), linestyle='dashed', marker='o', label='LR (O)')
    plt.title('Regression Output RMSE vs # corrupted rows')
    plt.xlabel('corrupted rows')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('result/RMSE_mean.png')
    plt.show()

    np.savetxt('result/rmse_rlr.out', RMSEs_rlr)
    np.savetxt('result/rmse_lr.out', RMSEs_lr)
    np.savetxt('result/rmse_lro.out', RMSEs_lro)


def solve_LR(X, y):

    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    return w


if __name__ == "__main__":
    msg = "Usage: python new_evaluation inverse_flip/boundary_flip/poison"
    if len(sys.argv) <= 1:
        print(msg)
    else:
        arg = sys.argv[1]
        if arg == 'inverse_flip':
            label_flipping(type='inverse')
        elif arg == 'boundary_flip':
            label_flipping(type='boundary')
        elif arg == 'poison':
            regression_new_poisoning()
        else:
            print(msg)






