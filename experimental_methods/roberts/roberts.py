#%%
import numpy as np
import scipy.interpolate
from scipy.interpolate import CubicSpline
from pathlib import Path
import os
import pandas as pd
import json
import matplotlib.pyplot as plt

# Physical constants
q_e = 1.6e-19
m_e = 9.1e-31
m_Xe = 131.293/6.022e26

# Directories
SCRIPT_DIR = Path(os.path.split(__file__)[0])
HOME = SCRIPT_DIR / ".." / ".."
RATE_COEFF_DIR = HOME / "rate_coeffs"
REF_DIR = HOME / "mcmc_reference" / "ref_3charge"
REF_SIM = REF_DIR / "jsons" / "output_3charge.json"

def spline_diff_samples(x, f, s=1., k=1):
    """Fits a smoothing spline to estimate derivatives
    of function f.

    Args:
        x (ndarray): position array: (n_posn,)
        f (ndarray): Function samples array: (n_posn,) 

    Returns:
        diff_f (ndarray): derivative of function samples (n_posn)
    """

    norm = np.max(f)
    f /= norm
    
    # Fit the smoothing spline with the given smoothing parameter
    spline = scipy.interpolate.splrep(x, f,s=s, k=k)

    # spline = interpolate.CubicSpline(x, f[:,ii], bc_type=((1, 0.0), (1, 0.0)))
    # This "bc" setting is "clamped": This means that the first derivative 
    # is constrained to be zero at both boundaries.
    
    # spline = interpolate.UnivariateSpline(x, f[:, ii], k=order)
    
    # Evaluate the derivative of the spline at the original x points
    diff_f = scipy.interpolate.splev(x, spline, der=1)
    
    
    f *= norm
    diff_f *= norm
        
    return diff_f

#%% 
# Load rate coefficients and create interpolants
# Ionization reactions

e_prod = np.array([1,2,3,1,2,1])

def load_iz_reaction(Z, Zprime):
    rxn_file = RATE_COEFF_DIR / f"ionization_Xe{Z}_Xe{Zprime}.dat"
    return pd.read_csv(rxn_file, skiprows=1, delimiter="\t") if rxn_file.exists() else None

k_energy = "Energy (eV)"
k_coeffs = "Rate coefficient (m^3/s)"

rate_coeffs = []
rate_coeffs_itp = []
Te_iz = []
k_iz_list = []
n_charge = 3
e_prod = []
rxn_Z = []
for Z in range(n_charge):
    row = []
    row_itp = []
    for Zprime in range(n_charge+1):
        K = load_iz_reaction(Z, Zprime)
        row.append(K)

        if K is not None:
            row_itp.append(CubicSpline(K[k_energy], K[k_coeffs]))
            Te_iz.append(K[k_energy] / 1.5)
            k_iz_list.append(K[k_coeffs])
            rxn_Z.append(Z)
            e_prod.append(Zprime - Z)
        else:
            row_itp.append(None)

    rate_coeffs.append(row)
    rate_coeffs_itp.append(row_itp)

n_react = len(Te_iz)
assert n_react == 6

for i in range(1, n_react):
    assert np.all(Te_iz[0] == Te_iz[i])

Te_iz = Te_iz[0]
k_iz = np.array(k_iz_list)

# Store rate coefficients for reactions
# react_i[Z1, Z2, :] is the rate coefficient for a reaction going from Z1 to Z2
# Lower triangular part of the matrix is the negative transpose of the upper part
react_i = np.zeros((n_charge+1,n_charge+1,len(Te_iz)))
react_i[0,1,:] = k_iz[0]
react_i[0,2,:] = k_iz[1]
react_i[0,3,:] = k_iz[2]
react_i[1,2,:] = k_iz[3]
react_i[1,3,:] = k_iz[4]
react_i[2,3,:] = k_iz[5]
# Add negative rates for loss of ions of charge Z1 when going from Z1 -> Z2
react_i -= react_i.transpose((1,0,2))

#%%

"""
Backwards solving function 
"""
def calc_inv_hall(z, ne, Te, ui, E, B, z_ext, Pe, nn_bc, ji_bc, je_bc, n_charge=3, s=1., un=150):
    
    n_pts = len(z)
    
    # initialize storage
    jez = np.zeros(n_pts)
    jiz = np.zeros((n_pts, n_charge))
    num_dens = np.zeros((n_pts, n_charge+1))

    del_nn = np.zeros(n_pts)
    del_jez = np.zeros(n_pts)
    del_jiz = np.zeros((n_pts, n_charge))

    # specify BC
    jez[-1] = je_bc 
    jiz[-1,:] = ji_bc 
    num_dens[-1, 0] = nn_bc 
    num_dens[-1,1] = jiz[-1,0] / (q_e*ui[-1])
    num_dens[-1,2] = jiz[-1,1] / (2*q_e*np.sqrt(2)*ui[-1])
    num_dens[-1,3] = jiz[-1,2] / (3*q_e*np.sqrt(3)*ui[-1])
    
    # needed manipulation of data
    del_z =  -z[1:n_pts] + z[0:n_pts-1] # multiply by -1 to account for BE method

    def calc_rhs(num_dens, ne, Te):
        # nn
        prod_tot = sum(np.interp(Te, Te_iz, k_iz[c]) for c in range(n_charge))
        del_nn = -(ne * num_dens[0] / un) * prod_tot

        # ji
        del_jiz = np.zeros(len(num_dens) - 1)
        for c in range(1,n_charge+1):
            prod_tot = 0.0
            for cc in range(1,n_charge+1):
                if cc == c: continue
                prod_tot += num_dens[cc] * np.interp(Te, Te_iz, react_i[c, cc, :])

            del_jiz[c-1] = c * q_e * ne * prod_tot

        # je
        prod_tot = sum(e_prod[r] * num_dens[rxn_Z[r]] * np.interp(Te, Te_iz, k_rxn) for (r, k_rxn) in enumerate(k_iz))
        del_jez = -q_e * ne * prod_tot 

        return del_nn, del_jiz, del_jez

    # loop backwards over points
    for i in range(0, n_pts-1):
        # compute the index
        true_idx = n_pts - 1 - i

        # Calculate dy/dx (predictor)
        del_nn[true_idx], del_jiz[true_idx, :], del_jez[true_idx] = calc_rhs(num_dens[true_idx, :], ne[true_idx], Te[true_idx])
        
        # Update state variable (predictor)
        jez[true_idx - 1] = jez[true_idx] + del_jez[true_idx]*del_z[true_idx-1]
        num_dens[true_idx - 1, 0] = num_dens[true_idx, 0] + del_nn[true_idx]*del_z[true_idx-1]

        for Z in range(n_charge):
            jiz[true_idx-1, Z] = jiz[true_idx, Z] + del_jiz[true_idx, Z]*del_z[true_idx-1]
            num_dens[true_idx-1, Z+1] = jiz[true_idx-1,Z] / ((Z+1)*q_e*np.sqrt(Z+1)*ui[true_idx-1])

        # Calculate dy/dx (corrector)
        del_nn_corr, del_jiz_corr, del_jez_corr = calc_rhs(num_dens[true_idx-1, :], ne[true_idx-1], Te[true_idx-1])

        # Update derivatives
        del_nn[true_idx] = 0.5 * (del_nn[true_idx] + del_nn_corr) 
        del_jez[true_idx] = 0.5 * (del_jez[true_idx] + del_jez_corr)
        del_jiz[true_idx,:] = 0.5 * (del_jiz[true_idx, :] + del_jiz_corr)

        # Update state variables (corrector)
        jez[true_idx - 1] = jez[true_idx] + del_jez[true_idx]*del_z[true_idx-1]
        num_dens[true_idx - 1, 0] = num_dens[true_idx, 0] + del_nn[true_idx]*del_z[true_idx-1]

        for Z in range(n_charge):
            jiz[true_idx-1, Z] = jiz[true_idx, Z] + del_jiz[true_idx, Z]*del_z[true_idx-1]
            num_dens[true_idx-1, Z+1] = jiz[true_idx-1,Z] / ((Z+1)*q_e*np.sqrt(Z+1)*ui[true_idx-1])
        
    """
    Compute Parameter
    """
    #for drifts, use Parker's calculation for now to avoid gradient
    #ExB drift
    u_ExB = E / B

    #diamagnetic drift

    #use Parker's spline for the grad of pe
    #Pe uses full ITS data to avoid odd gradients at the end 
    gradP = spline_diff_samples(z_ext, Pe, s=s, k=3) 
    #gradP = np.gradient(Pe, z_ext)
    gradP = gradP[0:n_pts]#truncate to real domain
    
    
    u_gradP = gradP / (q_e*ne*B)

    #uez
    uez = jez /(-q_e*ne)

    #inverse parameter 
    inv_Omega = -uez / (u_ExB + u_gradP)
    
    return inv_Omega, num_dens[:, 1:], num_dens[:, 0], uez, u_ExB, u_gradP, gradP

#%%

# Load properties from simulation outputs
with open(REF_SIM, "rb") as fp:
    sim_data = json.load(fp)

output = sim_data["output"]["average"]
ions = output["ions"]["Xe"]
neutrals = output["neutrals"]["Xe"]
z = np.array(output["z"])
Ez = np.array(output["E"])
Te = np.array(output["Tev"])
ne = np.array(output["ne"])
Br = np.array(output["B"])
wce = q_e * Br / m_e
A = np.array(output["channel_area"])
nu_an = np.array(output["nu_anom"])
nu_class = np.array(output["nu_class"])
nu_ei = np.array(output["nu_ei"])
nu_en = np.array(output["nu_ei"])
nu_e = nu_an + nu_class
nn = np.array(neutrals["n"])
un = np.array(neutrals["u"])[-1]
ui = np.array([ion["u"] for ion in ions])
ni = np.array([ion["n"] for ion in ions])
ue = np.array(output["ue"])
niui = np.array([ion["nu"] for ion in ions])
ji = np.array([(i+1) * q_e * nu for (i, nu) in enumerate(niui)])
ji_tot = np.sum(ji, axis=0)
current_fracs = ji[:, -1] / ji_tot[-1]
Id = output["discharge_current"]
eta_b = output["current_eff"]
eta_m = output["mass_eff"]
Z_max = len(current_fracs)
inv_hall = nu_an / wce

mdot_i = np.sum(niui[:, -1] * m_Xe * A[-1])
mdot_n = nn[-1] * m_Xe * un * A[-1]
eta_m = mdot_i / (mdot_n + mdot_i)

# %%
# Calculate downsteam boundary conditions
i_end = len(z)-1
z_end = z[i_end]
Zs = np.arange(1, Z_max+1)


# %%
# Subsample data to mimic experiment
z_fine = z[z>=0.025] 
z_coarse = np.array([0.017, 0.019000000000000003, 0.021, 0.023, 0.025, 0.027000000000000003, 0.029, 0.031, 0.035, 0.039, 0.043, 0.047, 0.055, 0.065, 0.07500000000000001])
#z_locs = np.array(coarse)
z_locs = np.array(z_fine)

def resample_spline(z_fine, quantity, z_coarse=z_locs, noise_std=0.0, relative=False):
    # Original spline interpolation
    spl = scipy.interpolate.make_interp_spline(z_fine, quantity, k=3)
    
    # Evaluate spline at coarser points
    noise = noise_std * np.random.standard_normal(len(z_coarse))

    if relative:
        quantity_coarse = spl(z_coarse) * (1 + noise)
    else:
        range = max(quantity) - min(quantity)
        quantity_coarse = spl(z_coarse) + range * noise

    # Return new coarse spline
    spl = scipy.interpolate.make_smoothing_spline(z_coarse, quantity_coarse, lam=0)
    return spl


#%% Solve for inv hall

def calc_hall_z(z_locs, s=0.01, noise_std=0.0, un=un):
    resample = lambda q, relative=False: resample_spline(z, q, z_locs, noise_std=noise_std, relative=relative)(z_locs)
    _ne = resample(ne, relative=True)
    _Te = resample(Te, relative=True)
    _Ez = resample(Ez, relative=True)
    _Br = resample(Br, relative=True)
    _ui = resample(ui[0], relative=True)
    _pe = q_e * _ne * _Te

    ji_tot_end = q_e * _ne[-1] * _ui[-1] * np.sum(current_fracs / np.sqrt(Zs))
    je_end = ji_tot_end * (1 / eta_b - 1)
    ji_end = current_fracs * ji_tot_end
    nn_end = ji_tot_end / (q_e * un) * (1 / eta_m - 1)
    ue_end = -je_end / (_ne[-1] * q_e)

    inv_omega, _ni, _nn, _uez, u_ExB, u_gradP, grad_pe = calc_inv_hall(
        z_locs, _ne, _Te, _ui, _Ez, _Br, z_locs, _pe,
        nn_end, ji_end, je_end, s=s, un=un
    )

    out = dict(
        z=z_locs, ne=_ne, Tev=_Te, E=_Ez,
        pe=_pe, Br=_Br, nn=_nn, ue=_uez, ue_end=ue_end,
        u_ExB=u_ExB, u_gradP=u_gradP, grad_pe=grad_pe,
        inv_hall=inv_omega
    )

    for Z in range(_ni.shape[1]):
        out[f"ni_{Z+1}"] = _ni[:, Z]
    for Z in range(_ni.shape[1]):
        out[f"ui_{Z+1}"] = np.sqrt(Z+1) * _ui

    return out

s = 0.01
noise_std = 0.025
results_fine = calc_hall_z(z_fine, s=s, noise_std=noise_std, un=un)
results_coarse = calc_hall_z(z_coarse, s=s, noise_std=noise_std, un=un)

#inv_omega_fine = inv_omega
# %%
fig, ax = plt.subplots(1,1)
key = "ni"
ax.set(yscale = 'log')
ax.plot(z, ni[0])
ax.plot(z, ni[1])
ax.plot(z, ni[2])
ax.plot(z_coarse, results_coarse["ni_1"], '--o', color="tab:blue")
ax.plot(z_coarse, results_coarse["ni_2"], '--o', color="tab:orange")
ax.plot(z_coarse, results_coarse["ni_3"], '--o', color="tab:green")
ax.plot(z_coarse, results_coarse["nn"])

plt.show()

# %%
# %%
# Save observations to CSV and toml
keys = set(["z", "ni_1", "ni_2", "ni_3", "ui_2", "ui_3", "inv_hall", "nn", "ue"])
fullstate = {k: v for (k, v) in results_coarse.items() if k in keys}

fullstate_df = pd.DataFrame(fullstate)
fullstate_df.to_csv(SCRIPT_DIR / "roberts.csv", index=False)

z_trunc = z[1:-1] if z[-1] == 0.08 else z
obs_fields = ["ui_1", "ne", "E", "Tev", "nn", "ue"]

def write_toml_file(filename, obs_fields):
    # Write TOML output
    with open(SCRIPT_DIR / filename, "w") as fp:
        config = sim_data["input"]["config"]
        directory = (REF_DIR / "normalized").absolute()
        print(f"# This file was autogenerated by {os.path.split(__file__)[-1]}", file=fp)
        print(f'base_sim = "{directory}"', file=fp)
        print("stddev = 1.0", file=fp)
        
        mdot = config["propellants"][0]["flow_rate_kg_s"]
        un = config["propellants"][0]["velocity_m_s"]
        Vcc = config["cathode_coupling_voltage"]
        Vd = config["discharge_voltage"]
        fb = 1.0

        print("\n[params]", file=fp)
        print(f"anode_mass_flow_rate_kg_s = {mdot}", file=fp)
        print(f"discharge_voltage_v = {Vd}", file=fp)
        print(f"magnetic_field_scale = {fb}", file=fp)
        print(f"cathode_coupling_voltage = {Vcc}", file=fp)
        print(f"neutral_velocity_m_s = {un}", file=fp)

        for field in obs_fields:
            print(file=fp)
            print(f"[fields.{field}]", file=fp)
            print(f"normalized = false", file=fp)

            if field == "nn":
                print(f"x = {[float(z_coarse[-1])]}", file=fp)
                print(f"y = {[float(results_coarse["nn"][-1])]}", file=fp)
            elif field == "ue":
                print(f"x = {[float(z_coarse[-1])]}", file=fp)
                print(f"y = {[float(results_coarse["ue_end"])]}", file=fp)
            else:
                print(f"x = {z_coarse.tolist()}", file=fp)
                print(f"y = {results_coarse[field].tolist()}", file=fp)

write_toml_file("observation.toml", obs_fields)
write_toml_file("observation_lif_only.toml", ["ui_1", "E"])

# %%
