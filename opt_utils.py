import numpy as np
import cvxpy as cp
import networkx as nx


# decompose a positive semidefinite matrix X = YY*
def decompose_psd(X, eig_tol=1e-9):
    n = X.shape[0]
    eigen_values, eigen_vectors = np.linalg.eigh(X)
    non_neg_eig_idx = [i for i in range(n) if eigen_values[i] >= eig_tol]
    return eigen_vectors[:, non_neg_eig_idx] @ np.diag(np.sqrt(eigen_values[non_neg_eig_idx]))


# normalize the rows of matrix X
def normalize_rows(X):
    return X / np.linalg.norm(X, axis=1)[:, np.newaxis]


def print_rank_data(X, tol=1e-9):
    print("The eigenvalues are")
    print(np.linalg.eigvalsh(X))
    print("Rank of matrix is %s with tolerance %s." % (np.linalg.matrix_rank(X, tol, hermitian=True), tol))


# project each entry of vector v into the annulus with inner radius min_r and outer radius max_r centered at 0
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


# round each row of Y to 1 or -1
def hyperplane_rounding(Y, cost, min_r=1, max_r=1, n_iter=100):
    min_cost = np.Inf
    best_x = None
    d = Y.shape[1]
    rng = np.random.default_rng()
    for i in range(n_iter):
        r = rng.standard_normal((d, 1))
        if np.isscalar(min_r) and np.isscalar(max_r):
            if min_r == 1 and max_r == 1:
                x = np.sign(Y @ r)
            else:
                x = proj_to_annulus(Y @ r, min_r, max_r)
        else:
            x = proj_to_annulus(Y @ r, min_r, max_r)

        cost_val = cost(x)
        if cost_val < min_cost:
            min_cost = cost_val
            best_x = x
    return min_cost, best_x


# round each row of Y to a complex number of unit modulus
def complex_hyperplane_rounding(Y, cost, min_r=1, max_r=1, n_iter=100):
    min_cost = np.Inf
    best_x = None
    d = Y.shape[1]
    rng = np.random.default_rng()
    for i in range(n_iter):
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
def eigen_proj(Y, is_complex, delta_rank=1):
    X = np.outer(Y, Y.conj())
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
def fixed_point_iteration(prob, X, shift, is_complex):
    n = X.shape[0]
    if is_complex:
        prev_X = cp.Parameter((n, n), hermitian=True, value=X.value)
        iteration_obj = cp.real(cp.trace(X @ (prev_X + shift)))
    else:
        prev_X = cp.Parameter((n, n), symmetric=True, value=X.value)
        iteration_obj = cp.trace(X @ (prev_X + shift))

    def mat_rank(X):
        return np.linalg.matrix_rank(X, tol=1e-9, hermitian=True)

    iteration_prob = cp.Problem(cp.Maximize(iteration_obj), prob.constraints)
    terminate = False
    print("Initial objective: ", prob.objective.value)
    print("Initial rank: ", mat_rank(X.value))
    while not terminate:
        iteration_prob.solve()
        if np.linalg.norm(X.value - prev_X.value) < 1e-6:
            terminate = True
        else:
            print("Current objective: ", prob.objective.value)
            print("Current rank: ", mat_rank(X.value))
            prev_X.value = X.value
    print("Fixed point objective: ", prob.objective.value)
    print("Fixed point rank: ", mat_rank(X.value))
    print("Fixed point eigenvalues:")
    print(np.linalg.eigvalsh(X.value))


# load a graph as a networkx Graph
def load_graph(graph_file, n):
    data_path = "../dat/"

    with open(data_path + graph_file) as inf:
        next(inf, '')  # skip first line
        G = nx.read_weighted_edgelist(inf, nodetype=int, encoding="utf-8")

    first_vertex = np.floor(np.random.default_rng().random() * (len(G) - n - 1)).astype(int)
    G = G.subgraph(range(first_vertex, first_vertex + n))
    assert len(G) == n
    nx.draw(G)
    return G


def build_enriched_supergraph(G, treewidth_algorithm_idx=0):
    # use the chosen algorithm to compute the approximate minimal tree decomposition
    treewidth_algorithms_list = [nx.algorithms.approximation.treewidth_min_degree,
                                 nx.algorithms.approximation.treewidth_min_fill_in]  # nx.junction_tree
    treewidth, tree_decomp = treewidth_algorithms_list[treewidth_algorithm_idx](G)
    print("Treewidth: %s" % treewidth)

    G_bar = G.copy()
    next_idx = G_bar.number_of_nodes()  # keep track of index of the next redundant vertex to be added
    T_bar = tree_decomp.copy()

    # add redundant vertices
    for bag in tree_decomp.nodes:
        if len(bag) < treewidth + 1:
            new_bag = bag.union(frozenset(range(next_idx, next_idx + treewidth + 1 - len(bag))))
            T_bar = nx.relabel_nodes(T_bar, {bag: new_bag})
            next_idx += treewidth + 1 - len(bag)
    G_bar.add_nodes_from(range(G_bar.number_of_nodes(), next_idx))

    # iterate through leaves of T_tilde and add edges
    T_tilde = T_bar.copy()
    while T_tilde.number_of_nodes() > 1:
        for bag in T_tilde.nodes:
            if T_tilde.degree(bag) == 1:  # the current bag is a leaf
                parent = list(T_tilde.neighbors(bag))[0]
                Os = sorted(list(bag.difference(parent)))
                Ws = sorted(list(parent.difference(bag)))
                G_bar.add_edges_from([(Os[i], Ws[i]) for i in range(len(Os))])
                T_tilde.remove_node(bag)
                break

    return G_bar
