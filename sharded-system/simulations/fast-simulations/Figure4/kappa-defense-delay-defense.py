#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan (ε_{j0}, α_{j0,in}) under three latency distributions and plot TWO figures:
1) Average (mean) κ vs (ε, α)
2) Maximum κ vs (ε, α)

Updates in this version:
- X-axis now shows ONLY 5 evenly spaced ticks (cleaner).
- Figures are compact; axis label & tick fonts enlarged.
- Keeps robust κ(t) calculation and self-consistent demand matrix.

Parameter sweep
---------------
ε: 4.0 → 16.0, step 0.5
α: mapped linearly to ε:
   α = 0.45 + (ε-4.0)/(16.0-4.0) * (0.95-0.45)

Self-consistent base_demand_matrix at each (ε, α):
   T = g_max(shard0) * L_target
   01 = 02 = 0
   10 = 20 = α * T / 2
   00 = 11 = 22 = (1 - α) * T
   12 = 21 = α * T / 4

We modify:
- demand.epsilon_matrix[1][0] = demand.epsilon_matrix[2][0] = ε
- delay.weights = one of:
    Spike   : [0, 0, 0, 0, 1]
    Uniform : [0.2, 0.2, 0.2, 0.2, 0.2]
    Bimodal : [0.4, 0, 0.2, 0, 0.4]

κ(t):
   κ(t) = Σ_d w_d * |ΔP(t-d)| / |ΔP(t)|
          (only if sign(ΔP(t-d)) == sign(ΔP(t)))
"""

import os, glob, time, shutil, subprocess
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------- global typography (bigger axis/labels/ticks) ----------
TITLE_FONTSIZE  = 16
LABEL_FONTSIZE  = 14
TICK_FONTSIZE   = 12
LEGEND_FONTSIZE = 11

mpl.rcParams.update({
    "axes.titlesize":  TITLE_FONTSIZE,
    "axes.labelsize":  LABEL_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
})

# ---------------- Sweep setup ----------------

EPS_MIN, EPS_MAX, EPS_STEP = 4, 16.0, 0.5
ALPHA_MIN, ALPHA_MAX       = 0.45, 0.95

EQUIL_P    = 1.0
START_STEP = 100
END_STEP   = 4999

DISTS = {
    "Spike"  : [0, 0, 0, 0, 1],
    "Uniform": [0.2, 0.2, 0.2, 0.2, 0.2],
    "Bimodal": [0.4, 0, 0.2, 0, 0.4],
}

# --------------- Config I/O helpers ---------------

def load_cfg(path='config.yml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_cfg(cfg, path='config.yml'):
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def backup_cfg():
    shutil.copy('config.yml', 'config_backup.yml')
    print("✅ backup config.yml")

def restore_cfg():
    if os.path.exists('config_backup.yml'):
        shutil.copy('config_backup.yml', 'config.yml')
        print("✅ restored config.yml")

# --------------- Model helpers --------------------

def target_T(cfg):
    l_target = float(cfg['simulation']['l_target'])
    g0 = None
    for s in cfg['shards']:
        if int(s.get('id', -1)) == 0:
            g0 = float(s['g_max'])
            break
    if g0 is None:
        raise ValueError("shards[0].g_max not found")
    return l_target * g0

def build_self_consistent_matrix(T, alpha):
    """3x3 base_demand_matrix that satisfies the shard totals & α constraint."""
    alpha = max(0.0, min(1.0, float(alpha)))
    g00 = (1 - alpha) * T
    g10 = g20 = 0.5 * alpha * T
    g01 = g02 = 0.0
    g11 = g22 = g00
    g12 = g21 = 0.25 * alpha * T
    return [
        [g00, g01, g02],
        [g10, g11, g12],
        [g20, g21, g22],
    ]

def set_eps_j0(cfg, eps):
    epsM = cfg['demand']['epsilon_matrix']
    epsM[1][0] = float(eps)
    epsM[2][0] = float(eps)

def set_base_matrix(cfg, M):
    cfg['demand']['base_demand_matrix'] = [[float(x) for x in row] for row in M]

def set_delay_weights(cfg, weights):
    cfg['delay']['weights'] = [float(w) for w in weights]

# --------------- Simulation & parsing --------------

def run_sim():
    r = subprocess.run(['go', 'run', '../../../main.go'],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print("❌ sim failed:", r.stderr[:400])
        return False
    time.sleep(0.05)
    return True

def parse_latest_log():
    files = glob.glob('enhanced_simulation_analysis_*.log')
    if not files:
        return None
    latest = max(files, key=os.path.getctime)
    data = []
    with open(latest, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            parts = line.split(',')
            if len(parts) < 3:
                continue
            try:
                step = int(parts[0])
                fee  = float(parts[1])
                load = float(parts[2])
                data.append({'Step': step, 'Fee': fee, 'Load': load})
            except:
                continue
    return pd.DataFrame(data) if data else None

# --------------- κ(t) calculation (robust) --------

def calc_kappa_timeseries(delta_p, weights, start_idx, end_idx):
    """
    κ(t) = Σ_d w_d * |ΔP(t-d)| / |ΔP(t)| with same-sign mask.
    Robust sign check; avoids prev*cur multiplication to prevent overflow.
    """
    dmax = len(weights)
    end_idx = min(end_idx, len(delta_p) - 1)
    if start_idx + dmax > end_idx:
        return []

    ks = []
    eps = 1e-12

    for t in range(start_idx + dmax, end_idx + 1):
        cur = delta_p[t]
        if not np.isfinite(cur) or abs(cur) < eps:
            continue

        s_cur = np.sign(cur)  # -1, 0, +1
        k = 0.0

        for d in range(1, dmax + 1):
            prev = delta_p[t - d]
            if not np.isfinite(prev) or prev == 0.0:
                continue
            # same-sign mask without multiplication to avoid overflow
            if np.sign(prev) == s_cur:
                k += weights[d - 1] * (abs(prev) / abs(cur))

        ks.append(k)

    return ks

# --------------- Sweep runner ---------------------

def alpha_from_eps(eps):
    """Map ε ∈ [4,16] to α ∈ [0.45,0.95] linearly."""
    return ALPHA_MIN + (eps - EPS_MIN) / (EPS_MAX - EPS_MIN) * (ALPHA_MAX - ALPHA_MIN)

def sweep_for_distribution(dist_name, weights, T):
    """Run the whole (ε, α) sweep for a given delay-weight shape."""
    eps_values = np.arange(EPS_MIN, EPS_MAX + 1e-9, EPS_STEP)
    alpha_values = np.array([alpha_from_eps(e) for e in eps_values])
    # tuple-style labels "(a, b)"
    labels_pairs = [f"({e:.1f}, {a:.3f})" for e, a in zip(eps_values, alpha_values)]

    means, maxes = [], []

    for e, a, lab in zip(eps_values, alpha_values, labels_pairs):
        print(f"\n[{dist_name}] {lab}")
        cfg = load_cfg()

        set_delay_weights(cfg, weights)
        set_eps_j0(cfg, e)
        M = build_self_consistent_matrix(T, a)
        set_base_matrix(cfg, M)
        save_cfg(cfg)

        if not run_sim():
            means.append(np.nan); maxes.append(np.nan)
            continue

        df = parse_latest_log()
        if df is None or df.empty:
            print("  ⚠️ no data parsed")
            means.append(np.nan); maxes.append(np.nan)
            continue

        if not (df['Step'] >= START_STEP).any() or not (df['Step'] <= END_STEP).any():
            print("  ⚠️ window not found")
            means.append(np.nan); maxes.append(np.nan)
            continue

        start_idx = df[df['Step'] >= START_STEP].index[0]
        end_idx   = df[df['Step'] <= END_STEP].index[-1]

        delta_p = (df['Fee'].values - EQUIL_P)
        ks = calc_kappa_timeseries(delta_p, weights, start_idx, end_idx)

        if ks:
            mean_k = float(np.mean(ks))
            max_k  = float(np.max(ks))
            print(f"  mean κ = {mean_k:.4f},  max κ = {max_k:.4f}")
        else:
            mean_k = np.nan; max_k = np.nan
            print("  κ series empty; mean/max = NaN")

        means.append(mean_k)
        maxes.append(max_k)

    # save per-distribution CSV
    out_rows = [
        {'distribution': dist_name,
         'epsilon_j0': float(e),
         'alpha_j0_in': float(a),
         'mean_kappa': (float(m) if m == m else None),
         'max_kappa' : (float(x) if x == x else None)}
        for e, a, m, x in zip(eps_values, alpha_values, means, maxes)
    ]
    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(f'kappa_scan_{dist_name}.csv', index=False)
    print(f"💾 saved kappa_scan_{dist_name}.csv")

    return eps_values, alpha_values, labels_pairs, np.array(means), np.array(maxes)

# --------------- Main -----------------------------

def main():
    backup_cfg()
    try:
        cfg0 = load_cfg()
        T = target_T(cfg0)

        # run for all three distributions
        results = {}
        for name, w in DISTS.items():
            eps_vals, alp_vals, labels_pairs, means, maxes = sweep_for_distribution(name, w, T)
            results[name] = {
                'eps': eps_vals, 'alpha': alp_vals, 'labels_pairs': labels_pairs,
                'mean': means, 'max': maxes
            }

        # save combined CSV
        combined = []
        for name, r in results.items():
            for e, a, m, x in zip(r['eps'], r['alpha'], r['mean'], r['max']):
                combined.append({
                    'distribution': name,
                    'epsilon_j0': float(e),
                    'alpha_j0_in': float(a),
                    'mean_kappa': (float(m) if m == m else None),
                    'max_kappa' : (float(x) if x == x else None),
                })
        pd.DataFrame(combined).to_csv('kappa_scan_all.csv', index=False)
        print("💾 saved kappa_scan_all.csv")

        # common x grid and tick selection (ONLY 5 evenly spaced x ticks)
        x = np.arange(len(results['Spike']['labels_pairs']))
        xtick_idx = np.linspace(0, len(x) - 1, 5).astype(int)
        xtick_labels = [results['Spike']['labels_pairs'][i] for i in xtick_idx]

        # ---- styles & helper ----
        style = {
            'Spike'  : {'marker': 'o'},
            'Uniform': {'marker': 's'},
            'Bimodal': {'marker': 'D'},
        }
        def prettify_axes(ax):
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
            for sp in ['bottom','left','top','right']:
                ax.spines[sp].set_linewidth(1.1)

        # 1) Mean κ figure  —— legend 固定左上角
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        for name in ['Spike', 'Uniform', 'Bimodal']:
            r = results[name]
            ax.plot(x, r['mean'], linestyle='-', marker=style[name]['marker'],
                    markersize=6, linewidth=1.7, label=name)  # only distribution name
        ax.set_xticks(xtick_idx)
        ax.set_xticklabels(xtick_labels, rotation=0)
        ax.set_xlabel('($ε_{j0}$, $\sumα_{j0,in}$}) pairs', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Mean κ', fontsize=LABEL_FONTSIZE)
        prettify_axes(ax)
        ax.legend(loc='upper left', frameon=True)  # ★ 不要再调用第二次 legend
        fig.tight_layout()
        fig.savefig('kappa_mean_curves.pdf')
        fig.savefig('kappa_mean_curves.png', dpi=240)
        print("📈 saved kappa_mean_curves.pdf / .png")

        # 2) Max κ figure —— 也放左上角
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        for name in ['Spike', 'Uniform', 'Bimodal']:
            r = results[name]
            ax.plot(x, r['max'], linestyle='--', marker=style[name]['marker'],
                    markersize=6, linewidth=1.7, label=name)  # only distribution name
        ax.set_xticks(xtick_idx)
        ax.set_xticklabels(xtick_labels, rotation=0)
        ax.set_xlabel('($ε_{j0}$, $\sumα_{j0,in}$}) pairs', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Max κ', fontsize=LABEL_FONTSIZE)
        prettify_axes(ax)
        ax.legend(loc='upper left', frameon=True)  # ★ 明确指定左上角
        fig.tight_layout()
        fig.savefig('kappa_max_curves.pdf')
        fig.savefig('kappa_max_curves.png', dpi=240)
        print("📈 saved kappa_max_curves.pdf / .png")

    finally:
        restore_cfg()

if __name__ == "__main__":
    main()
