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
import numpy as np
import scipy
import matplotlib.pyplot as plt


# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# %%
size = 50
step = 1
en = np.linspace(1, size, np.ceil(size / step).astype(int))

# %%
scint_scale_k = 2
scint_shift = 5

scint_eff = 1 - sigmoid((en - en[en.size // 2] - scint_shift) / scint_scale_k)

# scint_eff = np.ones(en.shape)

plt.plot(scint_eff)
plt.show()

# %%
intensity = np.sin(np.pi * en / en.max())
intensity /= intensity.sum()

plt.plot(intensity)
plt.show()

# %%
rng = np.random.default_rng()

n_of_filters = 25

filter_scale_k = rng.random(size=n_of_filters)
filter_scale_k *= size / 5

filter_shift = rng.random(size=n_of_filters)
filter_shift -= 0.5
filter_shift *= size / 1.25

print(filter_scale_k, filter_shift)

# %%
fig, ax = plt.subplots(1, 3, figsize=(21, 5))

coeff_array = []
scint_sums = []

for k, shift in zip(filter_scale_k, filter_shift):
    filter_tr = sigmoid((en - en[en.size // 2] - shift) / k)
    intensity_f = intensity * filter_tr * step
    intensity_fsc = intensity_f * scint_eff
    ax[0].plot(en, filter_tr)
    ax[1].plot(en, intensity_f)
    ax[2].plot(en, intensity_fsc)
    coeff_array.append(intensity_f)
    scint_sums.append(intensity_fsc.sum())

plt.show()

# filter_tr_1 = sigmoid((en - en[en.size // 2] - filter_shift[0]) / filter_scale_k[0])
# filter_tr_2 = sigmoid((en - en[en.size // 2] - filter_shift[1]) / filter_scale_k[1])
# intensity_f_1 = intensity * filter_tr_1 * step
# intensity_f_2 = intensity * filter_tr_2 * step
# intensity_fsc_1 = intensity_f_1 * scint_eff
# intensity_fsc_2 = intensity_f_2 * scint_eff

# %%
coeff_array = np.array(coeff_array)
scint_sums = np.array(scint_sums)

# data_array = np.array([en, intensity_f_1, intensity_f_2])
# scint_sum_1 = intensity_fsc_1.sum()
# scint_sum_2 = intensity_fsc_2.sum()
# print(data_array)

# %%
res = np.linalg.lstsq(coeff_array, scint_sums, rcond=None)
# res

# %%
plt.plot(res[0])
plt.plot(scint_eff)
plt.grid()
plt.show()

# %%

# %%

# %%
