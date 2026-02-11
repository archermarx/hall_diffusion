#%% 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy.interpolate import CubicSpline
import scipy.integrate
import os

#%% 
# Physical constants
q_e = 1.6e-19
m_e = 9.1e-31
m_Xe = 131.293/6.022e26

# Directories
SCRIPT_DIR = Path(os.path.split(__file__)[0])
HOME = Path(".")
RATE_COEFF_DIR = HOME / "rate_coeffs"
REF_DIR = HOME / "mcmc_reference" / "ref_3charge"
REF_SIM = REF_DIR / "jsons" / "output_3charge.json"

#%%
# Ionization reactions
def load_iz_reaction(Z, Zprime):
    rxn_file = RATE_COEFF_DIR / f"ionization_Xe{Z}_Xe{Zprime}.dat"
    return pd.read_csv(rxn_file, skiprows=1, delimiter="\t") if rxn_file.exists() else None

k_energy = "Energy (eV)"
k_coeffs = "Rate coefficient (m^3/s)"

#%% 
# Load rate coefficients and create interpolants
rate_coeffs = []
rate_coeffs_itp = []
for Z in range(3):
    row = []
    row_itp = []
    for Zprime in range(4):
        K = load_iz_reaction(Z, Zprime)
        row.append(K)

        if K is not None:
            row_itp.append(CubicSpline(K[k_energy], K[k_coeffs]))
        else:
            row_itp.append(None)

    rate_coeffs.append(row)
    rate_coeffs_itp.append(row_itp)

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

ji_tot_end = q_e * ne[i_end] * ui[0][i_end] * np.sum(current_fracs / np.sqrt(Zs))
je_end = ji_tot_end * (1 / eta_b - 1)
ji_end = current_fracs * ji_tot_end
nn_end = ji_tot_end / (q_e * un) * (1 / eta_m - 1)
bc = np.concat([[nn_end], [je_end], ji_end])

# %%
# Set up ODE system and poperty interpolants

spline_order=3
def Spline(x, y):
    return scipy.interpolate.make_interp_spline(x, y, k=spline_order)

ne_itp = Spline(z, ne) 
Te_itp = Spline(z, Te)
Ez_itp = Spline(z, Ez)
Br_itp = Spline(z, Br)
ui_itp = Spline(z, ui[0])

def ode_rhs(_z, state):
    nn = state[0]
    je = state[1]
    ji = state[2:]
    ui = np.sqrt(Zs) * ui_itp(_z)
    ni = ji / (q_e * ui)
    ne = ne_itp(_z)
    Te = Te_itp(_z)
    rhs = np.zeros_like(state)

    # Neutrals
    for Z_prime in range(1, Z_max+1):
        rhs[0] += rate_coeffs_itp[0][Z_prime](Te)

    # Electrons
    for Z in range(Z_max):
        for Z_prime in range(Z+1, Z_max+1):
            rhs[1] += (Z_prime - Z) * ni[Z] * rate_coeffs_itp[Z][Z_prime](Te)

    # Ions
    for Z in range(1, Z_max+1):
        # Production from lower charge states
        for Z_prime in range(Z):
            rhs[Z+1] += ni[Z_prime] * rate_coeffs_itp[Z_prime][Z](Te)

        # Depletion to higher charge states
        for Z_prime in range(Z+1, Z_max+1):
            rhs[Z+1] += ni[Z] * rate_coeffs_itp[Z][Z_prime](Te)

        
    rhs[0] *= -(ne * nn) / un
    rhs[1] *= -q_e * ne
    rhs[2:] *= Zs * q_e * ne
    return rhs
# %%
# Solve differential equation
zspan = (z_end, 0.020)

output = scipy.integrate.solve_ivp(ode_rhs, t_span=zspan, y0=bc, rtol=1e-10)

print(f'Success: {output["success"]}')
z_sol = output["t"]
f_sol = output["y"]

nn_sol = f_sol[0]
je_sol = f_sol[1]
ji_sol = f_sol[2:]
ui_sol = np.sqrt(Zs)[..., None] * ui_itp(z_sol)[None, ...]
ni_sol = ji_sol / ui_sol / q_e

ui_sol_km = ui_sol / 1000
ui_km = ui / 1000

# Electron drift speeds
pe_itp = Spline(z, ne * Te)
Ez_itp = Spline(z, Ez)
Br_itp = Spline(z, Br)
grad_pe_itp = pe_itp.derivative()

ne_sol = ne_itp(z_sol)

ue_ExB = Ez_itp(z_sol) / Br_itp(z_sol)
ue_gradpe = grad_pe_itp(z_sol)/(ne_sol * Br_itp(z_sol))
ue_drift = ue_ExB + ue_gradpe
uez_sol = -je_sol / (q_e * ne_sol)
inv_hall_sol = -uez_sol / ue_drift
# %%

fig, axes = plt.subplots(2, 3, figsize=(9,5.5), layout='constrained')
axes = axes.ravel()
z_sol_mm = z_sol * 1000
z_mm = z * 1000

common_args = dict(xlim=(z_mm[0], z_mm[-1]))
upper_args = dict(xticklabels=[])
lower_args = dict(xlabel = "z (mm)")
colors = ["tab:blue", "tab:orange", "tab:green"]
lw = 3

# nn
axes[0].set(yscale="log", title = "Neutral density (m$^{-3}$)", **upper_args)
axes[0].plot(z_mm, nn, label = "Original", linestyle='--', color=colors[0])
axes[0].plot(z_sol_mm, nn_sol, label = "Roberts", linewidth=lw, color=colors[0])
axes[0].legend()

for i in range(Z_max):
    # ni(1)
    axes[1].set(yscale="log", title = "Ion density (m$^{-3}$)", **upper_args)
    axes[1].plot(z_sol_mm, ni_sol[i], color=colors[i], linewidth=lw)
    axes[1].plot(z_mm, ni[i], linestyle='--', color=colors[i])

    # ui
    axes[2].set(title="Ion velocity (km/s)", **upper_args)
    axes[2].plot(z_sol_mm, ui_sol_km[i], color=colors[i], linewidth=lw)
    axes[2].plot(z_mm, ui_km[i], linestyle='--', color=colors[i])

# ue
axes[3].set(title="Axial electron velocity (km/s)", **lower_args)
axes[3].plot(z_sol_mm, uez_sol/1000, color=colors[0], linewidth=lw)
axes[3].plot(z_mm, ue/1000, color=colors[0], linestyle='--')

# inv hall
axes[4].set(title = "Inverse Hall parameter", yscale='log', **lower_args)
axes[4].plot(z_sol_mm, inv_hall_sol, color=colors[0], linewidth=lw)
axes[4].plot(z_mm, inv_hall, color=colors[0], linestyle='--')

fig.savefig(SCRIPT_DIR / "roberts_sim.png", dpi=200)
# %%
# Save observations to CSV and toml
fullstate = dict(
    z = z_sol,
    ue = uez_sol,
    inv_hall = inv_hall_sol,
    nn = nn_sol,
    Tev = Te_itp(z_sol),
    ne = ne_itp(z_sol),
    E = Ez_itp(z_sol),
)

for Z in range(Z_max):
    fullstate[f"ni_{Z+1}"] = ni_sol[Z]
    fullstate[f"ui_{Z+1}"] = ui_sol[Z]

fullstate_df = pd.DataFrame(fullstate)
fullstate_df.to_csv(SCRIPT_DIR / "roberts.csv", index=False)

z_trunc = z[1:-1] if z[-1] == 0.08 else z
z_inds = z_trunc >= zspan[-1]
z_locs = z_trunc[z_inds]
obs_fields = ["ui_1", "ne", "E", "Tev"]

# Write TOML output
with open(SCRIPT_DIR / "observation.toml", "w") as fp:
    directory = (REF_DIR / "normalized").absolute()
    print(f"# This file was autogenerated by {os.path.split(__file__)[-1]}", file=fp)
    print(f'base_sim = "{directory}"', file=fp)
    for field in obs_fields:
        print(file=fp)
        print(f"[fields.{field}]", file=fp)
        print(f"x = {z_locs.tolist()}", file=fp)

# %%
