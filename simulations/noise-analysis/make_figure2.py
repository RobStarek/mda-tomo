import numpy as np
import matplotlib.pyplot as plt
import h5py
import rounder
from IPython.display import display, Math, Markdown


with h5py.File('output_smp_noise_30.h5', 'r') as f:
    dset_smp = f['results_collection']
    arr_smp = np.array(dset_smp)
    shots = dset_smp.attrs['cnts']

with h5py.File('output_env_noise_30.h5', 'r') as f:
    dset_env = f['results_collection']
    arr_env = np.array(dset_env)
    mix_factors = dset_env.attrs['mix_factors']
    mix_factors_labels = dset_env.attrs['mix_factors_labels']

#from the main simulation
TH_LIMIT_OPTIM_P = (1-1e-3, -999, 1-1e-3+0.8e-3, 1-1e-3-1e-3)
TH_LIMIT_OPTIM_F = (1-3e-4, -999, 1-3e-4-5e-4, 1-3e-4+3e-4)

plt.rcParams.update({'errorbar.capsize': 2})
plt.rcParams['font.size'] = 9
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['figure.figsize'] = (6, 4)

fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
labels = ('naive', 'optimal', 'true')
colors = ['C0', 'C1', 'C2']

# --- ax1 and ax2: last point as hline+band ---
for i, label in enumerate(labels):
    x_vis_offset = (1 + i / 16)
    # All but last point as errorbar
    xvals = np.array(shots[:-1]) * x_vis_offset
    mu = 1 - arr_smp[:-1, i, 0]
    errhi = arr_smp[:-1, i, 3] - arr_smp[:-1, i, 0]
    errlo = arr_smp[:-1, i, 0] - arr_smp[:-1, i, 2]
    ax1.errorbar(xvals, mu, yerr=(errlo, errhi), fmt=".", label=label, color=colors[i])
    xlim = ax1.get_xlim()
    if i == 0:
        # Last point as hline+band
        y_inf = 1 - arr_smp[-1, i, 0]
        ylo_inf = 1 - arr_smp[-1, i, 3]
        yhi_inf = 1 - arr_smp[-1, i, 2]
    elif i == 2:
        y_inf = 1 - TH_LIMIT_OPTIM_P[0]
        ylo_inf = 1 - TH_LIMIT_OPTIM_P[3]
        yhi_inf = 1 - TH_LIMIT_OPTIM_P[2]
    if i==0 or i==2:
        ax1.axhline(y_inf, color=colors[i], linestyle='dashed')
        ax1.fill_between([shots[0], shots[-2]*10], ylo_inf, yhi_inf, color=colors[i], alpha=0.15, zorder=0)
ax1.set_xscale('log')
# ax1.legend()
ax1.grid()
ax1.set_ylabel('$1-\\min\\,P$')

xax = mix_factors[:,0]
for i, label in enumerate(labels):
    x_vis_offset = (1 + i / 16)
    xvals = np.array(shots[:-1]) * x_vis_offset
    mu = 1 - arr_smp[:-1, i+3, 0]
    errhi = arr_smp[:-1, i+3, 3] - arr_smp[:-1, i+3, 0]
    errlo = arr_smp[:-1, i+3, 0] - arr_smp[:-1, i+3, 2]
    ax2.errorbar(xvals, mu, yerr=(errlo, errhi), fmt=".", color=colors[i])
    # y_inf = 1 - arr_smp[-1, i+3, 0]
    # ylo_inf = 1 - arr_smp[-1, i+3, 3]
    # yhi_inf = 1 - arr_smp[-1, i+3, 2]
    # ax2.axhline(y_inf, color=colors[i], linestyle='dashed')
    # ax2.fill_between([shots[0], shots[-2]*10], ylo_inf, yhi_inf, color=colors[i], alpha=0.15, zorder=0)
    if i == 0:
        y_inf = 1 - arr_smp[0, i+3, 0]
        ylo_inf = 1 - arr_smp[0, i+3, 3]
        yhi_inf = 1 - arr_smp[0, i+3, 2]
    elif i == 2:
        y_inf = 1 - TH_LIMIT_OPTIM_F[0]
        ylo_inf = 1 - TH_LIMIT_OPTIM_F[3]
        yhi_inf = 1 - TH_LIMIT_OPTIM_F[2]
    if i==0 or i==2:    
        ax2.axhline(y_inf, color=colors[i], linestyle='dashed')
        ax2.fill_between([shots[0], shots[-2]*10], ylo_inf, yhi_inf, color=colors[i], alpha=0.15, zorder=0)

ax2.set_xscale('log')
ax2.set_xlabel('shots')
ax2.set_ylabel('$1-\\min F$')
ax2.grid()

# --- ax3 and ax4: first point as hline+band ---
xax = mix_factors[:,0]
xlab = ['1e-3', '1', '5', '10', '100']

for i, label in enumerate(('naive','true', 'optimal')):
    x_vis_offset = 0
    # All but first point as errorbar
    xvals = xax[1:] + x_vis_offset
    mu = 1 - arr_env[1:, i, 0]
    errhi = arr_env[1:, i, 3] - arr_env[1:, i, 0]
    errlo = arr_env[1:, i, 0] - arr_env[1:, i, 2]
    ax3.errorbar(xvals, mu, yerr=(errlo, errhi), fmt=".", color=colors[i], label=label)
    # First point as hline+band
    if i == 0:
        y0 = 1 - arr_env[0, i, 0]
        ylo0 = 1 - arr_env[0, i, 3]
        yhi0 = 1 - arr_env[0, i, 2]
    elif i == 2:
        y0 = 1 - TH_LIMIT_OPTIM_P[0]
        ylo0 = 1 - TH_LIMIT_OPTIM_P[3]
        yhi0 = 1 - TH_LIMIT_OPTIM_P[2]
    if i==0 or i==2:    
        ax3.axhline(y0, color=colors[i], linestyle='dashed')
        ax3.fill_between([xax[0]/2, xax[-1]*2], ylo0, yhi0, color=colors[i], alpha=0.15, zorder=0)
ax3.set_xscale('log')
ax3.grid()

for i, label in enumerate(('naive','true', 'optimal')):
    x_vis_offset = 0
    xvals = xax[1:] + x_vis_offset
    mu = 1 - arr_env[1:, i+3, 0]
    errhi = arr_env[1:, i+3, 3] - arr_env[1:, i+3, 0]
    errlo = arr_env[1:, i+3, 0] - arr_env[1:, i+3, 2]
    ax4.errorbar(xvals, mu, yerr=(errlo, errhi), fmt=".", color=colors[i])
    # y0 = 1 - arr_env[0, i+3, 0]
    # ylo0 = 1 - arr_env[0, i+3, 3]
    # yhi0 = 1 - arr_env[0, i+3, 2]
    # ax4.axhline(y0, color=colors[i], linestyle='dashed')
    # ax4.fill_between([xax[0]/2, xax[-1]*2], ylo0, yhi0, color=colors[i], alpha=0.15, zorder=0)
    # First point as hline+band
    if i == 0:
        y0 = 1 - arr_env[0, i+3, 0]
        ylo0 = 1 - arr_env[0, i+3, 3]
        yhi0 = 1 - arr_env[0, i+3, 2]
    elif i == 2:
        y0 = 1 - TH_LIMIT_OPTIM_F[0]
        ylo0 = 1 - TH_LIMIT_OPTIM_F[3]
        yhi0 = 1 - TH_LIMIT_OPTIM_F[2]
    if i==0 or i==2:    
        ax4.axhline(y0, color=colors[i], linestyle='dashed')
        ax4.fill_between([xax[0]/2, xax[-1]*2], ylo0, yhi0, color=colors[i], alpha=0.15, zorder=0)

ax4.set_xscale('log')
ax4.set_xlabel('$(p_x=p_y)$')
ax4.grid()

for axi in (ax1, ax2, ax3, ax4):
    axi.set_yscale('log')
for axi in (ax1, ax3):
    axi.set_ylim(1e-3 - 3e-4, None)
for axi in (ax2, ax4):
    axi.set_ylim(1e-4, None)

# Add panel labels (a, b, c, d) to the subplots
for ax, label in zip([ax1, ax2, ax3, ax4], ['a', 'b', 'c', 'd']):
    ax.annotate(f'({label})', xy=(0, 1.02), xycoords='axes fraction',
                ha='left', va='bottom')

plt.tight_layout()
plt.show()

## List value so it can be copied to the paper

display(Markdown("## Shot noise - impurity"))
for stat, line in zip(arr_smp, shots):
    for i, lab in enumerate('naive true optimized'.split()):
        mean = 1 - stat[i, 0]
        hi = 1 - stat[i, 2]
        lo = 1 - stat[i, 3]
        latex = rounder.FormatToError(mean, lo, hi)
        print(line, lab)
        print(latex)
        display(Math(latex))
    print("---------")


display(Markdown("## Shot noise - infidelity"))
for stat, line in zip(arr_smp, shots):
    for i, lab in enumerate('naive true optimized'.split()):
        mean = 1 - stat[i+3, 0]
        hi = 1 - stat[i+3, 2]
        lo = 1 - stat[i+3, 3]
        latex = rounder.FormatToError(mean, lo, hi)
        print(line, lab)
        print(latex)
        display(Math(latex))
    print("---------")

display(Markdown("## Env noise - impurity"))
for stat, line in zip(arr_env, mix_factors_labels):
    for i, lab in enumerate('naive true optimized'.split()):
        mean = 1 - stat[i, 0]
        hi = 1 - stat[i, 2]
        lo = 1 - stat[i, 3]
        latex = rounder.FormatToError(mean, lo, hi)
        print(line, lab)
        print(latex)
        display(Math(latex))
    print("---------")


display(Markdown("## Env noise - infidelity"))
for stat, line in zip(arr_env, mix_factors_labels):
    for i, lab in enumerate('naive true optimized'.split()):
        mean = 1 - stat[i+3, 0]
        hi = 1 - stat[i+3, 2]
        lo = 1 - stat[i+3, 3]
        latex = rounder.FormatToError(mean, lo, hi)
        print(line, lab)
        print(latex)
        print(mean)
        display(Math(latex))
    print("---------")
