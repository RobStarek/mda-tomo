import numpy as np
import matplotlib.pyplot as plt
import h5py

# Load data from HDF5 file
with h5py.File('output_smp_noise_30.h5', 'r') as f:
    dset_smp = f['results_collection']
    arr_smp = np.array(dset_smp)
    shots = dset_smp.attrs['cnts']
    metrics = dset_smp.attrs['metrics']
    statistics = dset_smp.attrs['statistics']

with h5py.File('output_env_noise_30.h5', 'r') as f:
    dset_env = f['results_collection']
    arr_env = np.array(dset_env)
    mix_factors = dset_env.attrs['mix_factors']
    mix_factors_labels = dset_env.attrs['mix_factors_labels']
    
# # Load data from HDF5 file
# with h5py.File('output_env_noise_30.h5', 'r') as f:
#     dset = f['results_collection']
#     results_collection = np.array(dset)
#     cnts = dset.attrs['cnts']
#     metrics = dset.attrs['metrics']
#     statistics = dset.attrs['statistics']

# Plotting
plt.rcParams.update({'errorbar.capsize': 2})
plt.rcParams['text.usetex'] = True #False
# Set the font family (you may adjust this to your preference)
#mpl.rcParams['font.family'] = 'serif'
#mpl.rcParams['font.serif'] = ['Times New Roman']
# set the font size to match the main text (adjust the value as needed)
plt.rcParams['font.size'] = 9
# Set the line width for the plot
plt.rcParams['lines.linewidth'] = 1.5
# Set the figure size for a single-column plot (adjust the width and height as needed)
plt.rcParams['figure.figsize'] = (5, 3)


fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
labels = ('naive', 'optimal', 'true')

# Plot minimum purity
for i, label in enumerate(labels):
    x_vis_offset = (1 + i / 16)
    mu = arr_smp[:, i, 0]
    errhi = arr_smp[:, i, 3] - mu
    errlo = mu - arr_smp[:, i, 2]
    ax1.errorbar(np.array(shots) * x_vis_offset, 1-mu, yerr=(errlo, errhi), fmt=".", label=label)
    ax1.set_xscale('log')
# ax1.legend()
ax1.grid()
ax1.set_ylabel('$1-\\min\\,P$')

# Plot minimum fidelity (dmae)
for i, label in enumerate(labels):
    x_vis_offset = (1 + i / 16)
    mu = arr_smp[:, i + 3, 0]
    errhi = arr_smp[:, i + 3, 3] - mu
    errlo = mu - arr_smp[:, i + 3, 2]
    ax2.errorbar(np.array(shots) * x_vis_offset, 1-mu, yerr=(errlo, errhi), fmt=".", label=label)
    ax2.set_xscale('log')
ax2.set_xlabel('shots')
ax2.set_ylabel('$1-\\min F$')
ax2.grid()


# pur/fid, nai/tru/opt, cal/test, mean/std
xax = mix_factors[:,0]
xax[0] = 1e-4
xlab = ['1e-3', '1', '5', '10', '100']

for i, label in enumerate(('naive','true', 'optimal')):
    x_vis_offset = 0
    mu = arr_env[:, i, 0]
    errhi = arr_env[:,i,3] - mu
    errlo = mu - arr_env[:,i,2]
    ax3.errorbar(x = xax+x_vis_offset, y = 1-mu, yerr = (errlo, errhi), fmt=".", label=label)
    ax3.set_xscale('log')

ax3.grid()
# # ax3.set_ylabel('$1 - \min\,P$')

for i, label in enumerate(('naive','true', 'optimal')):
    x_vis_offset = 0
    mu = arr_env[:, i+3, 0]
    errhi = arr_env[:,i+3,3] - mu
    errlo = mu - arr_env[:,i+3,2]
    ax4.errorbar(x = xax+x_vis_offset, y = 1-mu, yerr = (errlo, errhi), fmt=".", label=label)
    ax2.set_xscale('log')
    
# ax2.legend()
# ax4.set_xticks(xax, labels=xlab, rotation='vertical')
ax4.set_xlabel('$p_x=p_z$')
ax4.grid()

for axi in (ax1, ax2, ax3, ax4):
    axi.set_yscale('log')
    axi.set_ylim(1e-3, None)

# Add panel labels (a, b, c, d) to the subplots
for ax, label in zip([ax1, ax2, ax3, ax4], ['a', 'b', 'c', 'd']):
    ax.annotate(f'({label})', xy=(0, 1.02), xycoords='axes fraction',
                ha='left', va='bottom')
    
plt.tight_layout()
plt.show()
