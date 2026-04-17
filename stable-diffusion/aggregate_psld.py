"""
PSLD aggregation script — runs after all array tasks complete (no GPU needed).
Loads sample_*.pt tensors, computes posterior statistics, saves diagnostic plots.
"""
import os, glob, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_rgb(t):
    """[-1,1] tensor [C,H,W] → [0,1] numpy [H,W,C]."""
    return ((t.float().clamp(-1, 1) + 1) / 2).numpy().transpose(1, 2, 0)


def _mean_ch(t):
    """[C,H,W] tensor → mean over C, numpy [H,W]."""
    return t.float().mean(dim=0).numpy()


def _save_rgb(arr, path, title=''):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.clip(arr, 0, 1))
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def _save_std(t, path, title='Posterior Std Dev'):
    arr = _mean_ch(t)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(arr, cmap='hot', vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def _save_residual(t, path, title=''):
    arr = _mean_ch(t)
    vmax = float(max(abs(arr.min()), abs(arr.max()))) + 1e-8
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(arr, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def _save_summary(true_d, label_d, sample_d, mean_d, std_t,
                  r_x1, r_y1, r_xm, r_ym, path, n):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'PSLD posterior diagnostics  (N={n})', fontsize=13)

    def rgb(ax, arr, ttl):
        ax.imshow(np.clip(arr, 0, 1))
        ax.set_title(ttl, fontsize=10)
        ax.axis('off')

    def div(ax, t, ttl):
        arr = _mean_ch(t)
        vmax = float(max(abs(arr.min()), abs(arr.max()))) + 1e-8
        im = ax.imshow(arr, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(ttl, fontsize=10)
        ax.axis('off')

    def std(ax, t, ttl):
        arr = _mean_ch(t)
        im = ax.imshow(arr, cmap='hot', vmin=0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(ttl, fontsize=10)
        ax.axis('off')

    rgb(axes[0, 0], true_d,   'Ground Truth')
    rgb(axes[0, 1], label_d,  'Measurement y')
    rgb(axes[0, 2], sample_d, 'One Posterior Sample')
    rgb(axes[1, 0], mean_d,   'Posterior Mean')
    std(axes[1, 1], std_t,    f'Posterior Std Dev (N={n})')
    div(axes[1, 2], r_x1,     'GT − One Sample')
    div(axes[2, 0], r_y1,     'y − A(One Sample)')
    div(axes[2, 1], r_xm,     'GT − Posterior Mean')
    div(axes[2, 2], r_ym,     'y − A(Posterior Mean)')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',    type=str, required=True)
parser.add_argument('--n_posterior', type=int, default=100)
parser.add_argument('--image_tag',   type=str, default='apt')
args = parser.parse_args()

save_dir    = args.save_dir
samples_dir = os.path.join(save_dir, 'samples')
shared_dir  = os.path.join(save_dir, 'shared')
tag         = args.image_tag

for d in ['posterior_mean', 'posterior_std', 'residuals', 'summary', 'input', 'label', 'recon']:
    os.makedirs(os.path.join(save_dir, d), exist_ok=True)

# Load shared tensors saved by task 0
gt   = torch.load(os.path.join(shared_dir, 'gt.pt'),   weights_only=True)   # [C,H,W] in [-1,1]
y_n  = torch.load(os.path.join(shared_dir, 'yn.pt'),   weights_only=True)   # [C,H,W] in [-1,1]
mask = torch.load(os.path.join(shared_dir, 'mask.pt'), weights_only=True)   # [1,H,W] in {0,1}

# Load all posterior samples
sample_files = sorted(glob.glob(os.path.join(samples_dir, 'sample_*.pt')))
n_found = len(sample_files)
print(f"Found {n_found} / {args.n_posterior} samples.")
if n_found == 0:
    raise FileNotFoundError(f"No sample_*.pt files found in {samples_dir}")
if n_found < args.n_posterior:
    print(f"WARNING: expected {args.n_posterior}, found {n_found}. Some tasks may have failed.")

print("Loading samples...")
samples = torch.stack([torch.load(f, weights_only=True) for f in sample_files])  # [N,C,H,W]

# Posterior statistics
post_mean = samples.mean(dim=0)  # [C,H,W]
post_std  = samples.std(dim=0)   # [C,H,W]
sample_0  = samples[0]           # [C,H,W]

def A(x):
    return x * mask  # mask [1,H,W] broadcasts over C

# Residuals (all in [-1,1] tensor space, [C,H,W])
r_x1 = gt  - sample_0          # GT − one sample
r_y1 = y_n - A(sample_0)       # y  − A(one sample)
r_xm = gt  - post_mean         # GT − posterior mean
r_ym = y_n - A(post_mean)      # y  − A(posterior mean)

# Display arrays (all [0,1] numpy [H,W,C])
true_d   = _to_rgb(gt)
sample_d = _to_rgb(sample_0)
mean_d   = _to_rgb(post_mean)
label_d  = _to_rgb(y_n * mask + (-1.0) * (1.0 - mask))  # masked region → black

# --- Save individual outputs ---
_save_rgb(true_d,   os.path.join(save_dir, 'input',          f'{tag}_clean.png'),    'Ground Truth')
_save_rgb(label_d,  os.path.join(save_dir, 'label',          f'{tag}_degraded.png'), 'Measurement y')
_save_rgb(sample_d, os.path.join(save_dir, 'recon',          f'{tag}_sample0.png'),  'One Posterior Sample')
_save_rgb(mean_d,   os.path.join(save_dir, 'posterior_mean', f'{tag}_mean.png'),     'Posterior Mean')

_save_std(post_std,
          os.path.join(save_dir, 'posterior_std', f'{tag}_std.png'),
          title=f'Posterior Std Dev  (N={n_found})')
_save_std(post_std,
          os.path.join(save_dir, 'posterior_std', f'{tag}_std_5x.png'),
          title=f'Posterior Std Dev ×5  (N={n_found})')

res_dir = os.path.join(save_dir, 'residuals')
_save_residual(r_x1, os.path.join(res_dir, f'{tag}_res_x_sample0.png'), 'GT − One Sample')
_save_residual(r_y1, os.path.join(res_dir, f'{tag}_res_y_sample0.png'), 'y − A(One Sample)')
_save_residual(r_xm, os.path.join(res_dir, f'{tag}_res_x_mean.png'),    'GT − Posterior Mean')
_save_residual(r_ym, os.path.join(res_dir, f'{tag}_res_y_mean.png'),    'y − A(Posterior Mean)')

_save_summary(true_d, label_d, sample_d, mean_d, post_std,
              r_x1, r_y1, r_xm, r_ym,
              os.path.join(save_dir, 'summary', f'{tag}_summary.png'),
              n=n_found)

psnr_one  = psnr(true_d, sample_d)
psnr_mean = psnr(true_d, mean_d)
print(f"PSNR (one sample):     {psnr_one:.2f} dB")
print(f"PSNR (posterior mean): {psnr_mean:.2f} dB")
print(f"Mean std:              {post_std.mean():.4f}")
print(f"\nDone. Results in: {save_dir}/")
