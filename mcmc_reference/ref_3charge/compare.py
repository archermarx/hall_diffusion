import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def get_fields(sim_file):
    with open(sim_file, "rb") as fp:
        sim = json.load(fp)
    
    print(sim["output"].keys())
    avg = sim["output"]["average"]
    ions = avg["ions"]["Xe"]

    z = np.array(avg["z"])
    n = np.array(avg["ne"])
    u = np.array(ions[0]["u"])
    T = np.array(avg["Tev"])
    t = np.array([frame["t"] for frame in sim["output"]["frames"]])
    Id = np.array([frame["discharge_current"] for frame in sim["output"]["frames"]])

    return z, n, u, T, t, Id

dir_mcmc = Path("mcmc_reference/ref_3charge/")
z1, n1, u1, T1, t1, Id1 = get_fields(dir_mcmc / "ref_3charge.json")
z2, n2, u2, T2, t2, Id2 = get_fields(dir_mcmc / "output_3charge.json")

print(f"{len(z1)=}, {len(z2)=}")

fig, axes = plt.subplots(2,2, layout="constrained", figsize=(9,9))
axes = axes.ravel()

axes[0].plot(z1, n1, label = "New")
axes[0].plot(z2, n2, label = "Old")
axes[0].legend()

axes[1].plot(z1, u1, label = "New")
axes[1].plot(z2, u2, label = "Old")

axes[2].plot(z1, T1, label = "New")
axes[2].plot(z2, T2, label = "Old")

axes[3].plot(t1, Id1)
axes[3].plot(t2, Id2)



fig.savefig(dir_mcmc / "comparison.png", dpi=200)
