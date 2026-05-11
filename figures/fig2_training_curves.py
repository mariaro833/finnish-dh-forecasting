
import json

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt



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



curves = json.load(open('training_curves.json'))



configs = [

    ('v2_abs',   '#2166ac', 'TFT-abs v2 (absolute target)'),

    ('v3_resid', '#d6604d', 'TFT-resid v3 (residual target)'),

]



fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)



for ax, (label, color, title) in zip(axes, configs):

    d = curves[label]

    train_steps = [e['step'] for e in d['train_loss_epoch']]

    train_vals  = [e['value'] for e in d['train_loss_epoch']]

    val_steps   = [e['step'] for e in d['val_loss']]

    val_vals    = [e['value'] for e in d['val_loss']]



    ax.plot(train_steps, train_vals, color=color, linestyle='-', label='Train loss', alpha=0.9)

    ax.plot(val_steps, val_vals, color=color, linestyle='--', label='Val loss', alpha=0.9)



    best_val = min(val_vals)

    best_ep  = val_steps[val_vals.index(best_val)]

    ax.axvline(best_ep, color='black', linestyle=':', linewidth=1.0, alpha=0.6)

    ax.annotate(f'best\n{best_val:.1f}', xy=(best_ep, best_val),

                xytext=(best_ep + 0.6, best_val * 1.04),

                fontsize=8, color='black')



    ax.set_title(title)

    ax.set_xlabel('Training step')

    ax.set_ylabel('Quantile loss')

    ax.legend(loc='upper right', framealpha=0.9)



plt.tight_layout()

plt.savefig('fig2_training_curves.pdf', bbox_inches='tight')

plt.savefig('fig2_training_curves.png', dpi=200, bbox_inches='tight')

print("Figure 2 saved.")

