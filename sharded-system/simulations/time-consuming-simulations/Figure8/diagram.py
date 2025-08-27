#!/usr/bin/env python3
# Plot Ri vs Gi with native 'o' / 'x' markers (unchanged look),
# log-scaled Gi (ticks labeled as 10^k), and theoretical boundary.
# Only the scatter points are rasterized (high DPI) to keep PDF small.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ======== paths ========
CSV_PATH = "phase_map_validation_Spike.csv"         # <-- replace with your CSV
OUT_PDF  = "phase_map_prediction.pdf"

# ======== style: larger fonts, no title ========
plt.rcParams.update({
    "font.size": 18,          # ~ +8pt vs default
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "pdf.compression": 9,
})

# ======== load data ========
df = pd.read_csv(CSV_PATH)

for col in ["Gi", "Ri"]:
    if col not in df.columns:
        raise ValueError(f"CSV must contain column '{col}'")

df["Gi"] = pd.to_numeric(df["Gi"], errors="coerce")
df["Ri"] = pd.to_numeric(df["Ri"], errors="coerce")

# Empirical convergence: prefer boolean column 'converged', fallback to 'final_load'
if "converged" in df.columns:
    def to_bool(x):
        if isinstance(x, bool): return x
        if isinstance(x, (int, np.integer)): return x != 0
        if isinstance(x, str): return x.strip().lower() in {"true","1","yes","y","t"}
        return False
    df["empirical"] = df["converged"].apply(to_bool)
elif "final_load" in df.columns:
    df["final_load"] = pd.to_numeric(df["final_load"], errors="coerce")
    df["empirical"] = (df["final_load"] - 0.5).abs() < 1e-5
else:
    raise ValueError("Need 'converged' or 'final_load' in CSV to derive empirical convergence")

# Theoretical stability region: Ri < 1 and Gi*(1+Ri) < 2
df["theory"] = (df["Ri"] < 1) & ((df["Gi"] * (1 + df["Ri"])) < 2)

# Consistency
df = df[np.isfinite(df["Gi"]) & np.isfinite(df["Ri"]) & (df["Gi"] > 0)].copy()
df["consistent"] = (df["empirical"] == df["theory"])

greens = df[df["consistent"]]    # consistent → green circle
reds   = df[~df["consistent"]]   # inconsistent → red cross
n_green, n_red = len(greens), len(reds)

# ======== plot ========
fig, ax = plt.subplots(figsize=(7, 5.4))

# Gi on log scale, ticks shown as 10^k
ax.set_yscale('log')
ax.yaxis.set_major_locator(mtick.LogLocator(base=10.0))
ax.yaxis.set_major_formatter(mtick.LogFormatterMathtext(base=10.0))

# X limits with small padding
x_min, x_max = np.nanmin(df["Ri"]), np.nanmax(df["Ri"])
x_pad = 0.02 * (x_max - x_min if x_max > x_min else 1.0)
ax.set_xlim(x_min - x_pad, x_max + x_pad)

# --- scatter with native markers; rasterize only these collections ---
# Use high savefig dpi so rasterized points are crisp when zooming
scatter_kwargs = dict(s=20, alpha=0.9, rasterized=True)

if n_green > 0:
    ax.scatter(greens["Ri"], greens["Gi"], marker='o',
               c="#2ca02c", edgecolors='none', **scatter_kwargs, label=None)
if n_red > 0:
    ax.scatter(reds["Ri"], reds["Gi"], marker='x',
               c="#d62728", linewidths=1.2, **scatter_kwargs, label=None)

# --- theoretical boundary ---
ax.axvline(1.0, color="black", lw=2, ls="--", label="Ri = 1")

ri = np.linspace(max(1e-6, x_min - x_pad), x_max + x_pad, 1000)
gi = 2.0 / (1.0 + ri)
gi[gi <= 0] = np.nan
ax.plot(ri, gi, color="blue", lw=2, label="Gi = 2/(1+Ri)")

# Labels (English), legend upper-left with counts
ax.set_xlabel("Ri (coupling ratio)")
ax.set_ylabel("Gi (intensity)")

from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0],[0], marker='o', color='none', markerfacecolor="#2ca02c",
           markeredgecolor='none', markersize=10, label=f"Consistent ({n_green})"),
    Line2D([0],[0], marker='x', color="#d62728", markersize=10,
           markeredgewidth=1.2, label=f"Inconsistent ({n_red})"),
    Line2D([0],[0], color="blue", lw=2, label="Gi = 2/(1+Ri)"),
    Line2D([0],[0], color="black", lw=2, ls="--", label="Ri = 1"),
]
ax.legend(handles=legend_handles, loc="upper left", frameon=False)

ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_PDF, dpi=600)   # high DPI → crisp rasterized points
print(f"Saved: {OUT_PDF}  (consistent: {n_green}, inconsistent: {n_red})")
