import pandapower as pp
import pandapower.networks as pn


def get_case(case_name, case_type=None):
    # case type is specified only for custom cases
    if case_type is not None:
        return globals()[case_name](case_type)
    else:
        net = getattr(pn, case_name)()
        return net


# 2-bus simple case
def ccase2(case_type=0):
    net = pp.create_empty_network(sn_mva=1)

    # buses
    pp.create_bus(net, vn_kv=1, max_vm_pu=1.1, min_vm_pu=0.9, index=0)
    pp.create_bus(net, vn_kv=1, max_vm_pu=1.1, min_vm_pu=0.9, index=1)

    #lines
    pp.create_line_from_parameters(net, 0, 1, length_km=1, r_ohm_per_km=1, x_ohm_per_km=1, c_nf_per_km=0, max_i_ka=0, index=0)

    # generators
    pp.create_ext_grid(net, 0, max_p_mw=1, min_p_mw=0, max_q_mvar=1, min_q_mvar=-1, index=0)
    pp.create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=1, cp0_eur=0, cq1_eur_per_mvar=0, cq0_eur=0, cp2_eur_per_mw2=0, cq2_eur_per_mvar2=0, index=None)

    # loads
    pp.create_load(net, 1, p_mw=0.01, q_mvar=0, index=0)

    return net


# 3-bus radial simple case
def ccase3_radial(case_type=0):
    net = pp.create_empty_network(sn_mva=1)

    # buses
    pp.create_bus(net, vn_kv=1, max_vm_pu=1.1, min_vm_pu=0.9, index=0)
    pp.create_bus(net, vn_kv=1, max_vm_pu=1.1, min_vm_pu=0.9, index=1)
    pp.create_bus(net, vn_kv=1, max_vm_pu=1.1, min_vm_pu=0.9, index=2)

    #lines
    if case_type == 0:
        pp.create_line_from_parameters(net, 0, 1, length_km=1, r_ohm_per_km=1, x_ohm_per_km=1, c_nf_per_km=0, max_i_ka=0, index=0)
        pp.create_line_from_parameters(net, 2, 0, length_km=1, r_ohm_per_km=1, x_ohm_per_km=1, c_nf_per_km=0, max_i_ka=0, index=1)
    elif case_type == 1:
        pp.create_line_from_parameters(net, 0, 1, length_km=1, r_ohm_per_km=5, x_ohm_per_km=2, c_nf_per_km=0, max_i_ka=0, index=0)
        pp.create_line_from_parameters(net, 2, 0, length_km=1, r_ohm_per_km=2, x_ohm_per_km=4, c_nf_per_km=0, max_i_ka=0, index=1)

    # generators
    pp.create_ext_grid(net, 0, max_p_mw=1, min_p_mw=0, max_q_mvar=1, min_q_mvar=-1, index=0)
    pp.create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=1, cp0_eur=0, cq1_eur_per_mvar=0, cq0_eur=0, cp2_eur_per_mw2=0, cq2_eur_per_mvar2=0, index=None)

    # loads
    pp.create_load(net, 1, p_mw=0.01, q_mvar=0, index=0)
    pp.create_load(net, 2, p_mw=0.01, q_mvar=0, index=1)

    return net



def ccase3(case_type=0):
    net = ccase3_radial(case_type)
    pp.create_line_from_parameters(net, 1, 2, length_km=1, r_ohm_per_km=1, x_ohm_per_km=1, c_nf_per_km=0, max_i_ka=0, index=2)

    return net
