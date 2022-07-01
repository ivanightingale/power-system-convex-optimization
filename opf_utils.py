import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
from pandapower.plotting import simple_plot


def load_pp_network(case):
    # net = custom_networks.get_case(case)
    net = getattr(pn, case)()
    n = len(net.bus)

    simple_plot(net, plot_loads=True, plot_sgens=True)

    # generators
    gen_df_list = []
    gen_class_list = ["ext_grid", "gen", "sgen"]  # the 3 types of generators
    data_col_list = ["bus", "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"]

    for gen_class in gen_class_list:
        if not net[gen_class].empty:
            # get a table of cost coefficients only for the current type of generators
            gen_class_poly_cost = net.poly_cost.loc[net.poly_cost.et == gen_class].set_index("element")
            # get a table of cost coefficients and power bounds only for the current type of generators
            gen_class_df = net[gen_class][data_col_list].join(gen_class_poly_cost)
            gen_df_list.append(gen_class_df)

    # combine tables for all types of generators
    gen_df = pd.concat(gen_df_list).reset_index()
    n_gen = len(gen_df)
    gens = gen_df["bus"].to_numpy()

    # loads
    load_df = net.bus.join(net.load[["bus", "p_mw", "q_mvar"]].set_index("bus")).fillna(0)[["p_mw", "q_mvar"]]

    # admittance
    # compute line susceptance in p.u.
    net.line['s_pu'] = net.line['c_nf_per_km'] * net.line["length_km"] * (2 * np.pi * net.f_hz) * \
                       ((net.bus.loc[net.line.from_bus.values, "vn_kv"].values) ** 2) / net.sn_mva / net.line['parallel'] / 1e9
    # obtain a NetworkX Graph from the network, with each edge containing p.u. impedance data
    graph = pp.topology.create_nxgraph(net, multi=False, calc_branch_impedances=True, branch_impedance_unit="pu")
    G_val, B_val, Y_val = compute_admittance_mat(net, graph)

    return net, n, gen_df, n_gen, gens, load_df, graph, G_val, B_val, Y_val



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


def recover_solution(X, proj_rank=1):
    eigen_values, eigen_vectors = np.linalg.eigh(X)
    eigen_values = np.real(eigen_values)
    X_proj = eigen_vectors[:, -proj_rank:] @ np.diag(eigen_values[-proj_rank:]) @ eigen_vectors[:, -proj_rank:].conj().T
    return X_proj


def recover_verify_solution(prob, X, proj_rank=1):
    X_tmp = X.value
    X.value = recover_solution(X.value, proj_rank)
    for c in prob.constraints:
        print(c.violation())
    X.value = X_tmp



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


# given a solution X, compute the power generation and check if they lie in bounds
def check_power_feasibility(X, p_min, p_max, q_min, q_max, gens, n_gen, p_d, q_d, G, B, graph):
    p_g, q_g = compute_generation(X, n_gen, p_d, q_d, G, B, graph)
    for i in range(n_gen):
        if p_g[i] < p_min[i]:
            print("active power generation %d is too low: p_g = %f < %f" % (gens[i], p_g[i], p_min[i]))
        elif p_g[i] > p_max[i]:
            print("active power generation %d is too high: p_g = %f > %f" % (gens[i], p_g[i], p_max[i]))
        if q_g[i] < q_min[i]:
            print("reactive power generation %d is too low: q_g = %f < %f" % (gens[i], q_g[i], q_min[i]))
        elif q_g[i] > q_max[i]:
            print("reactive power generation %d is too high: q_g = %f > %f" % (gens[i], q_g[i], q_max[i]))
