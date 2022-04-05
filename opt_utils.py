import numpy as np
import cvxpy as cp


# decompose a positive semidefinite matrix X = YY*
def decompose_psd(X):
    n = X.shape[0]
    eigen_values, eigen_vectors = np.linalg.eigh(X)
    non_neg_eig_idx = [i for i in range(n) if eigen_values[i] >= 0]
    return eigen_vectors[:, non_neg_eig_idx] @ np.diag(np.sqrt(eigen_values[non_neg_eig_idx]))


# normalize the rows of matrix X
def normalize_rows(X):
    return X / np.linalg.norm(X, axis=1)[:, np.newaxis]


# round each row of Y to 1 or -1
def hyperplane_rounding(Y, cost, iter=100):
    min_cost = np.Inf
    best_x = None
    d = Y.shape[1]
    rng = np.random.default_rng()
    for i in range(iter):
        r = rng.standard_normal((d, 1))
        x = np.sign(Y @ r)
        cost_val = cost(x)
        if cost_val < min_cost:
            min_cost = cost_val
            best_x = x
    return min_cost, best_x


# project each entry of complex vector v into the annulus with inner radius min_r and outer radius max_r centered at 0
def proj_to_annulus(v, min_r, max_r):
    # find the closest point of each entry in the ring
    n = v.shape[0]
    x = []
    norms = np.linalg.norm(v, axis=1)

    for i in range(n):
        v_i = v[i]
        x_i = None
        min_r_i = min_r if np.isscalar(min_r) else min_r[i]
        max_r_i = max_r if np.isscalar(max_r) else max_r[i]

        if norms[i] < min_r_i:
            x_i = v_i / norms[i] * min_r_i
        elif norms[i] > max_r_i:
            x_i = v_i / norms[i] * max_r_i
        else:
            x_i = v_i
        x += [x_i]

    return np.array(x)


# round each row of Y to a complex number of unit modulus
def complex_hyperplane_rounding(Y, cost, min_r=1, max_r=1, iter=100):
    min_cost = np.Inf
    best_x = None
    d = Y.shape[1]
    rng = np.random.default_rng()
    for i in range(iter):
        r = 1 / np.sqrt(2) * rng.standard_normal((d, 1)) + 1 / np.sqrt(2) * rng.standard_normal((d, 1)) * 1j
        if np.isscalar(min_r) and np.isscalar(max_r):
            if min_r == 1 and max_r == 1:
                x = normalize_rows(Y @ r)
            else:
                x = proj_to_annulus(Y @ r, min_r, max_r)
        else:
            x = proj_to_annulus(Y @ r, min_r, max_r)
        cost_val = cost(x)
        if cost_val < min_cost:
            min_cost = cost_val
            best_x = x
    return min_cost, best_x


# approximately reduce rank of X = YY* by delta_rank via eigenprojection and row normalization
# if can't be reduced anymore, return the original Y
def eigen_proj(Y, delta_rank, is_complex):
    X = Y @ Y.conj().T
    current_rank = np.linalg.matrix_rank(X, tol=1e-9)
    target_rank = current_rank - delta_rank
    Y_proj = Y
    new_rank = current_rank
    if target_rank >= 1:
        new_rank = target_rank
        eigen_values, eigen_vectors = np.linalg.eigh(X)
        eigen_values = np.real(eigen_values)
        if not is_complex:
            eigen_vectors = np.real(eigen_vectors)
        # sort by eigenvalues, from the largest to the smallest
        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]

        Y_proj = eigen_vectors[:, range(new_rank)] @ np.diag(np.sqrt(eigen_values[range(new_rank)]))
        Y_proj = normalize_rows(Y_proj)
    return Y_proj, new_rank


# perform fixed point iteration on cvxpy variable X, where X is the optimal solution of prob
def fixed_point_iteration(prob, X, is_complex):
    if is_complex:
        X_val = cp.Parameter((X.shape[0], X.shape[1]), hermitian=True, value=X.value)
        iteration_obj = cp.real(cp.trace(X @ X_val))
    else:
        X_val = cp.Parameter((X.shape[0], X.shape[1]), symmetric=True, value=X.value)
        iteration_obj = cp.trace(X @ X_val)

    def mat_rank(X):
        return np.linalg.matrix_rank(X, tol=1e-9)

    prev_X_val = X.value
    iteration_prob = cp.Problem(cp.Maximize(iteration_obj), prob.constraints)
    terminate = False
    print("Initial rank: ", mat_rank(X.value))
    print("Initial objective: ", prob.objective.value)
    while not terminate:
        iteration_prob.solve()
        print("Current objective: ", prob.objective.value)
        print("Current rank: ", mat_rank(X.value))
        X_val.value = X.value
        terminate = np.linalg.norm(X.value - prev_X_val) < 1e-6
        prev_X_val = X.value
    print("Fixed point rank: ", mat_rank(X.value))
    print("Fixed point objective: ", prob.objective.value)
    print("Fixed point eigenvalues:")
    print(np.linalg.eigvalsh(X.value))
