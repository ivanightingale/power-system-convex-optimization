import numpy as np


# compute power generation (p_d, q_d) given X = vv*
def compute_generation(X, n_gen, p_d, q_d, G, B, graph):
    p_g = np.zeros(n_gen)
    q_g = np.zeros(n_gen)
    for i in range(n_gen):
        p_g[i] = p_d[i][0] + G[i][i] * X[i][i] + np.sum(
            [G[i][j] * X[i][j].real + B[i][j] * X[i][j].imag for j in graph.neighbors(i)])
        q_g[i] = q_d[i][0] - B[i][i] * X[i][i] + np.sum(
            [-B[i][j] * X[i][j].real + G[i][j] * X[i][j].imag for j in graph.neighbors(i)])
    return np.real(p_g), np.real(q_g)


# given a solution v, compute the power generation and check if they lie in bounds
def check_power_feasibility(v, p_min, p_max, q_min, q_max, n_gen, p_d, q_d, G, B, graph):
    X = v @ v.conj().T
    p_g, q_g = compute_generation(X, n_gen, p_d, q_d, G, B, graph)
    for i in range(n_gen):
        if p_g[i] < p_min[i]:
            print("active power generation %d is too low: p_g = %f < %f" % (i, p_g[i], p_min[i]))
        elif p_g[i] > p_max[i]:
            print("active power generation %d is too high: p_g = %f > %f" % (i, p_g[i], p_max[i]))
        if q_g[i] < q_min[i]:
            print("reactive power generation %d is too low: q_g = %f < %f" % (i, q_g[i], q_min[i]))
        elif q_g[i] > q_max[i]:
            print("reactive power generation %d is too high: q_g = %f > %f" % (i, q_g[i], q_max[i]))


# check whether the generation constratins are satisfied by a given solution (v, p_g, q_g)
def verify_feasibility(X, p_g, q_g, p_d, q_d, gen_df, G, B, graph):
    n = X.shape[0]
    for i in range(n):
        gen_list = gen_df.loc[gen_df["bus"] == i].index.to_numpy()

        active_lhs = np.sum([p_g[k] for k in gen_list]) - p_d[i]
        active_rhs = G[i][i] * X[i][i] + np.sum(
            [G[i][j] * np.real(X[i][j]) + B[i][j] * np.imag(X[i][j]) for j in graph.neighbors(i)])
        if np.abs(np.real(active_lhs - active_rhs)) > 1e-6:
            print("active %d is violated" % (i))

        reactive_lhs = np.sum([q_g[k] for k in gen_list]) - q_d[i]
        reactive_rhs = -B[i][i] * X[i][i] + np.sum(
            [-B[i][j] * np.real(X[i][j]) + G[i][j] * np.imag(X[i][j]) for j in graph.neighbors(i)])
        if np.abs(np.real(active_lhs - active_rhs)) > 1e-6:
            print("reactive %d is violated" % (i))
