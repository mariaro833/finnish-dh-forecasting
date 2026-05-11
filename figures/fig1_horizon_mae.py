
import json

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import numpy as np



plt.rcParams.update({

    'font.family': 'DejaVu Sans',

    'font.size': 10,

    'axes.titlesize': 11,

    'axes.labelsize': 10,

    'xtick.labelsize': 9,

    'ytick.labelsize': 9,

    'legend.fontsize': 9,

    'figure.dpi': 150,

    'axes.spines.top': False,

    'axes.spines.right': False,

    'axes.grid': True,

    'grid.alpha': 0.35,

    'grid.linestyle': '--',

    'grid.linewidth': 0.6,

    'lines.linewidth': 1.6,

})



xgb   = json.load(open('results_xgb_v3.json'))

tft   = json.load(open('results_v3.json'))

naive = json.load(open('results_naive.json'))



horizons  = np.arange(1, 25)

pers_mae  = naive['persistence_from_base']['mae_per_horizon']

seas_mae  = naive['seasonal_naive_24h']['mae_per_horizon']

xgb_mae   = xgb['mae_per_horizon']

tft_mae   = tft['mae_per_horizon']



pers_mean = float(np.mean(pers_mae))

seas_mean = float(np.mean(seas_mae))

xgb_mean  = float(np.mean(xgb_mae))

tft_mean  = float(np.mean(tft_mae))



fig, ax = plt.subplots(figsize=(7.5, 4.2))



ax.plot(horizons, pers_mae, color='#888888', linestyle='--', label=f'Naive persistence  (mean {pers_mean:.1f})', zorder=2)

ax.plot(horizons, seas_mae, color='#aaaaaa', linestyle=':', label=f'Seasonal naive 24h  (mean {seas_mean:.1f})', zorder=2)

ax.plot(horizons, xgb_mae,  color='#2166ac', linestyle='-', marker='o', markersize=3.5, label=f'XGB-resid v3  (mean {xgb_mean:.1f})', zorder=3)

ax.plot(horizons, tft_mae,  color='#d6604d', linestyle='-', marker='s', markersize=3.5, label=f'TFT-resid v3  (mean {tft_mean:.1f})', zorder=4)



ax.set_xlabel('Forecast horizon (h)')

ax.set_ylabel('MAE (MWh/h)')

ax.set_title('Per-horizon mean absolute error — test year 2024')

ax.set_xticks(horizons)

ax.xaxis.set_minor_locator(ticker.NullLocator())

ax.set_xlim(0.5, 24.5)

ax.set_ylim(0, max(seas_mae) * 1.12)

ax.legend(loc='lower right', framealpha=0.9)



plt.tight_layout()

plt.savefig('fig1_horizon_mae.pdf', bbox_inches='tight')

plt.savefig('fig1_horizon_mae.png', dpi=200, bbox_inches='tight')

print("Figure 1 saved.")

