# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os
import scipy.integrate, scipy.differentiate, scipy.interpolate

# %%
@dataclass
class VDF:
    v: np.ndarray
    g: np.ndarray
    z: float

# %%
def read_vdfs(folder):
    vdfs: list[VDF] = []

    for file in os.listdir(folder):
        fullpath = os.path.join(folder, file)
        df = pd.read_csv(fullpath)
        u = df.iloc[:, 0].to_numpy().squeeze() * 1000
        I = df.iloc[:, 1].to_numpy().squeeze()
        integral = scipy.integrate.trapezoid(I, u)
        g = I / integral

        z = float(file.split(".")[0].split("_")[1])
        vdfs.append(VDF(u, g, z))

    vdfs.sort(key = lambda x: x.z)

    return vdfs


# %%
vdfs = read_vdfs("data_35uTorr")

fig, ax = plt.subplots(1,1)

for vdf in vdfs:
    ax.plot(vdf.v, vdf.g, label = f"z = {vdf.z} mm")

#ax.legend()
plt.show()

# %%
def moment(vdf, order):
    integrand = (vdf.v**order) * vdf.g
    return scipy.integrate.trapezoid(integrand, vdf.v)

#def perez_luna(vdfs):
z = np.array([v.z / 1000 for v in vdfs])

u1 = np.array([moment(vdf, 1) for vdf in vdfs])
u2 = np.array([moment(vdf, 2) for vdf in vdfs])
u3 = np.array([moment(vdf, 3) for vdf in vdfs])
w1 = u1
w2 = u2/u1
w3 = u3/u2

q = 1.6e-19
M = 131.293 / 6.022e26

w2_spline = scipy.interpolate.splrep(z, w2)
w3_spline = scipy.interpolate.splrep(z, w3)
w2_spline_eval = scipy.interpolate.splev(z, w2_spline, der=0)
dw2_dz = scipy.interpolate.splev(z, w2_spline, der=1)
dw3_dz = scipy.interpolate.splev(z, w3_spline, der=1)

a_z = (w1 * w2) / (2 * w1 - w3) * dw3_dz
E_z = M / q * a_z
nu_iz = np.abs(a_z / w2 - (w1 / w2) * dw2_dz)

u_mp = [v.v[np.argmax(v.g)] for v in vdfs]

# %%
row_height = 3.5
col_width = 3
fig, axs = plt.subplots(1,3, layout='constrained', figsize=(3 * col_width, row_height))

axs[0].set(xlim=(0, 0.08/0.025))
axs[1].set(xlim=(0, 0.08/0.025))
axs[2].set(xlim=(0, 0.08/0.025))
z_lch = z / 0.025 + 1
axs[0].plot(z_lch, u_mp)
axs[1].plot(z_lch, E_z/1000)
axs[2].plot(z_lch, nu_iz)

# %%
out_dict = {"z": z + 0.025, "ui_1": u_mp, "E": E_z, "nu_iz": nu_iz}
df = pd.DataFrame(out_dict)
df.to_csv("perez_luna.csv", index=False)

# %%
