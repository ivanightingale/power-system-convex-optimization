import numpy as np



def compute_admittance_mat(net, graph):
    n = graph.number_of_nodes()
    # compute non-diagonal entries of the admittance matrices (opposite of mutual admittance on each line)
    G_val = np.zeros((n, n))
    B_val = np.zeros((n, n))
    for i,j in graph.edges:
        edge = graph.edges[(i,j)]
        r = edge["r_pu"]
        x = edge["x_pu"]
        z = r + x*1j
        y = np.reciprocal(z)
        G_val[i][j] = G_val[j][i] = -np.real(y)
        B_val[i][j] = B_val[j][i] = -np.imag(y)
    # compute intermediate values that will sum to the diagonal entries of G and B
    G_interm = -np.copy(G_val)
    B_interm = -np.copy(B_val)

    # add line susceptance
    for _, row in net.line.iterrows():
        B_interm[row["from_bus"]][row["to_bus"]] += row["s_pu"] / 2
        B_interm[row["to_bus"]][row["from_bus"]] += row["s_pu"] / 2

    # transformers
    for _, row in net.trafo.iterrows():
        h = row["hv_bus"]
        l = row["lv_bus"]
        ratio_magnitude = (row["vn_hv_kv"] / row["vn_lv_kv"]) * (net.bus.loc[l]["vn_kv"] / net.bus.loc[h]["vn_kv"])
        ratio_sq = ratio_magnitude**2
        G_interm[h][l] /= ratio_sq
        B_interm[h][l] /= ratio_sq
        G_interm[l][h] *= ratio_sq
        B_interm[l][h] *= ratio_sq
        # shift degrees
        if row["tap_phase_shifter"]:
            theta = row["shift_degree"]
            ratios_denom = np.exp(-1j * (2 * theta / 180 * np.pi))
            G_val[h][l] /= ratios_denom
            B_val[h][l] /= ratios_denom
            G_val[l][h] *= ratios_denom
            B_val[l][h] *= ratios_denom

    G_row_sums = np.sum(G_interm, axis=1)
    B_row_sums = np.sum(B_interm, axis=1)
    for i in range(n):
        shunt_row = net.shunt.loc[net.shunt["bus"] == i]
        g = 0  # shunt conductance
        b = 0  # shunt susceptance
        # if both are 0, there will be no data in the shunt dataframe
        if not shunt_row.empty:
            g = shunt_row["p_mw"] / net.sn_mva
            b = -shunt_row["q_mvar"] / net.sn_mva  # the pandapower data store negative shunt susceptance
        G_val[i][i] = g + G_row_sums[i]
        B_val[i][i] = b + B_row_sums[i]

    Y_val = G_val + B_val * 1j
    return G_val, B_val, Y_val

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
    X = np.outer(v, v.conj())
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
