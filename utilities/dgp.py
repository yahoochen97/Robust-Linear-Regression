import numpy as np

def generate_gaussian(SIZE, k):
    '''
    Generate independent gaussian.
    '''
    it = 0
    while it<10:
        m = np.random.normal(loc=0, scale=1, size=SIZE)
        if np.linalg.matrix_rank(m) == k:
            return m
        it += 1
    if it==10:
        print("Matrix has rank larger than k. Try again.")
        return None

def generate_pristine_data(n, k, m):
    '''
    Generate pristine data.
    '''
    # generate U
    U = generate_gaussian((n,k), k)

    # generate B
    B = generate_gaussian((k,m), k)
    
    if U is None or B is None:
        return None

    X_star = U.dot(B)
    return X_star, U
    

def poison_subspace_recovery(X_star, n1, k, m):
    '''
    Generate adversarial data.
    '''
    # generate U_A
    U_A = generate_gaussian((n1,k), k)

    # generate B_A
    B_A = generate_gaussian((k,m), k)
    
    if U_A is None or B_A is None:
        return None

    it = 0
    while it<10:
        idx = np.random.choice(X_star.shape[0], int(k/2), replace=False)
        B_A[:int(k/2),:] = X_star[idx]
        if np.linalg.matrix_rank(B_A) == k:
            break
        it += 1

    X_A = U_A.dot(B_A)
    return X_A


