import numpy as np
from utilities.robust_subspace_recovery import RSR
from utilities.dgp import generate_pristine_data, poison_subspace_recovery

def main():
    n = 350
    m = 50
    n1 = 50
    for k in range(20):
        X_star = generate_pristine_data(n, k, m)
        X_A = poison_subspace_recovery(X_star, n1, k, m)

if __name__ == "__main__":
    main()
