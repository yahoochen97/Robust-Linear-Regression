import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from utilities.robust_subspace_recovery import RSR
from utilities.robust_regression import RobustRegression
from utilities.poison_subspace_recovery import generate_pristine_data, poison_subspace_recovery
from utilities.poison_linear_regression import poison_linear_regression
from sklearn.preprocessing import normalize


def runtime():
    n = 350
    m = 400
    n1 = 50
    runs = 100
    times = []
    ks = range(1,21)
    for k in ks:
        ts = np.zeros((runs,))
        for run in range(runs):
            X_star, _ = generate_pristine_data(n-n1, k, m, iters=100)
            # poisoning for subspace recovery
            X_a1 = poison_subspace_recovery(X_star, n1, k, m)
            X_all = np.concatenate([X_star, X_a1], axis=0)
            robust_recovery = RSR(X_all, n-n1, n1, k, max_iters=100)
            start = time.time()
            _, _, _ = robust_recovery.recover()
            end = time.time()
            ts[run] = end - start
        # p25 = np.percentile(ts, 25)
        # p75 = np.percentile(ts, 75)
        # ts = ts[(ts<=p75) & (ts>=p25)]
        # print(ts)
        times.append(np.mean(ts))

    plt.plot(ks, times, label='TPCR')
    plt.xlabel('Rank')
    plt.ylabel('Time(s)')
    plt.legend(loc='best')
    plt.xticks([5*i for i in range(5)], [5*i for i in range(5)])
    plt.savefig("results/runtime.png")
    plt.show()

def recovery():
    m = 400
    k = 10
    n1s = [10+10*i for i in range(20)]
    irs = []
    for n1 in n1s:
        n = 400 - n1
        X_star, U_star = generate_pristine_data(n, k, m, iters=100)
        # poisoning for subspace recovery
        X_a1 = poison_subspace_recovery(X_star, n1, k, m)
        X_all = np.concatenate([X_star, X_a1], axis=0)
        robust_recovery = RSR(X_all, n, n1, k, max_iters=100)
        assignments, U, B = robust_recovery.recover()
        identification_rate = np.sum(assignments[-n1:]==0)/n1
        # diff = X_star-U[assignments.reshape(-1,)].dot(B.T)
        # print(np.sqrt(np.mean(diff**2)))
        irs.append(identification_rate)
    
    plt.plot(n1s, irs, label='TPCR')
    plt.xlabel('# Corrupted Rows')
    plt.ylabel('Identification Rate')
    plt.legend(loc='best')
    plt.xticks([50*i for i in range(5)], [50*i for i in range(5)])
    plt.show()

def regression():
    # poisoning for linear regression
    n = 380
    m = 20
    k = 20
    n1 = 400 - n
    X_star, U_star = generate_pristine_data(n, k, m, iters=1000)

    w_star = np.random.rand(m)
    y_star = X_star.dot(w_star)
    ind_adv_seeds = np.random.choice(n, n1, replace=False)
    X_c = X_star[ind_adv_seeds]
    y_c = y_star[ind_adv_seeds]
    lam = 0.001
    betta = 0.5
    sigma = 0.01
    eps = 0.01

    X_a2 = poison_linear_regression(X_star, y_star, X_c, y_c, lam, betta, sigma, eps)
    # X_a2 = (X_a2 - np.mean(X_a2)) / np.std(X_a2)
    print(np.mean(X_a2))
    print(np.std(X_a2))
    X_all = np.concatenate([X_star, X_a2], axis=0)
    y_all = np.concatenate([y_star, y_c], axis=0)

    # TODO: Trimmed Regression
    robust_regression = RobustRegression(X_all, y_all, n, n1, k, max_iters=100)
    w_predict = robust_regression.trimmed_principal_component_regression(X_star)
    y_predict = X_star.dot(w_predict)
    rmse = np.sqrt(np.mean(pow(y_predict - y_star, 2)))
    print(rmse)

    # robust_recovery = RSR(X_all, n, n1, k, max_iters=100)
    # assignments, U, B = robust_recovery.recover()

if __name__ == "__main__":
    msg = "Usage: python main.py runtime/recovery/regression"
    if len(sys.argv) <=1 :
        print(msg)
    else:
        arg = sys.argv[1]
        if arg == 'runtime':
            runtime()
        elif arg == 'recovery':
            recovery()
        elif arg == 'regression':
            regression()
        else:
            print(msg)
    
