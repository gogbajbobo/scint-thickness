# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext jupyter_black

# %%
import xraydb
import spekpy
import numpy as np
import matplotlib.pyplot as plt

# %%
en_step = 1

s = spekpy.Spek(
    physics="spekpy-v1",
    kvp=120,
    dk=en_step,
    targ="W",
    char=False,
    # shift=0.5,
)
# s.filter("Air", 100)
energies_keV, intensities = s.get_spectrum(
    flu=False,
    diff=True,
)
energies_eV = energies_keV * 1000

# %%
plt.semilogy(energies_keV, intensities)
plt.grid()
plt.show()

# %%
xraydb.add_material("CsI", "CsI", 4.51)

# %%
csi_mus = xraydb.material_mu("CsI", energies_eV)
csi_width_cm = 0.0125
csi_absorption = 1 - np.exp(-csi_mus * csi_width_cm)

plt.plot(energies_keV, csi_absorption)
plt.grid()
plt.show()

# %%
al_mus = xraydb.material_mu("Al", energies_eV)
# al_widths_cm = np.append(0, np.geomspace(1e-2, 1e1, 10))
al_widths_cm = np.linspace(0, 4, 4001)
al_transmissions = np.exp(-np.outer(al_mus, al_widths_cm).T)
al_intensities = al_transmissions * intensities
csi_al_intensities = al_intensities * csi_absorption

# %%
fig, ax = plt.subplots(1, 3, figsize=(21, 5))
plot_step = 100

for al_t in al_transmissions[::plot_step]:
    ax[0].plot(energies_keV, al_t)
ax[0].grid()

for al_intens in al_intensities[::plot_step]:
    ax[1].semilogy(energies_keV, al_intens)
ax[1].grid()
ax[1].set_ylim(1e6, 1e10)

for csi_al_intens in csi_al_intensities[::plot_step]:
    ax[2].semilogy(energies_keV, csi_al_intens)
ax[2].grid()
ax[2].set_ylim(1e6, 1e10)

plt.show()

# %%
csi_sums = csi_al_intensities.sum(axis=-1)

# %%
res = np.linalg.lstsq(al_intensities, csi_sums, rcond=None)

# %%
plt.plot(res[0])
plt.plot(csi_absorption)
plt.grid()
plt.show()

# %%

# %%

# %%
