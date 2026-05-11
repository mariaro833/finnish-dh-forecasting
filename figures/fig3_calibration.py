
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np



plt.rcParams.update({

    'font.family': 'DejaVu Sans',

    'font.size': 10,

    'axes.titlesize': 11,

    'axes.labelsize': 10,

    'xtick.labelsize': 10,

    'ytick.labelsize': 9,

    'legend.fontsize': 9,

    'figure.dpi': 150,

    'axes.spines.top': False,

    'axes.spines.right': False,

    'axes.grid': True,

    'grid.alpha': 0.35,

    'grid.linestyle': '--',

    'grid.linewidth': 0.6,

})



models   = ['XGB-abs v1', 'XGB-resid v3', 'TFT-resid v3']

coverage = [43.9, 74.0, 49.9]

colors   = ['#92c5de', '#2166ac', '#d6604d']



fig, ax = plt.subplots(figsize=(5.5, 4.0))



x = np.arange(len(models))

bars = ax.bar(x, coverage, color=colors, width=0.5, zorder=3, edgecolor='white', linewidth=0.8)



ax.axhline(80, color='black', linestyle='--', linewidth=1.4, label='Nominal 80% target', zorder=4)



for bar, val in zip(bars, coverage):

    ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2,

            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')



ax.set_xticks(x)

ax.set_xticklabels(models)

ax.set_ylabel('Empirical PI80 coverage (%)')

ax.set_title('80% prediction interval coverage by model')

ax.set_ylim(0, 92)

ax.legend(loc='upper right', framealpha=0.9)



plt.tight_layout()

plt.savefig('fig3_calibration.pdf', bbox_inches='tight')

plt.savefig('fig3_calibration.png', dpi=200, bbox_inches='tight')

print("Figure 3 saved.")

