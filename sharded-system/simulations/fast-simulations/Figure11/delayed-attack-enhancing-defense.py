#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scan delayed attack region (lambda_{j0}, alpha_{j0,in}) but MEASURE DEFENSE kappa (shard 0)
under three latency distributions.

Sweep:
  λ: 4.0 → 16.0 (step 0.5)
  α: 0.45 → 0.95 (linearly mapped from λ)

At each (λ, α):
  - Set lambda_matrix[1][0] = lambda_matrix[2][0] = λ   (λ_{10} = λ_{20})
  - Build self-consistent base_demand_matrix for INBOUND α to shard 0:
        T = g_max(shard0) * L_target
        10 = 20 = 0.5 * α * T
        01 = 02 = 0
        00 = 11 = 22 = (1 - α) * T
        12 = 21 = 0.25 * α * T
  - Set delay.weights to one of:
        Spike   : [0, 0, 0, 0, 1]
        Uniform : [0.2, 0.2, 0.2, 0.2, 0.2]
        Bimodal : [0.4, 0, 0.2, 0, 0.4]
  - Run sim and compute ONLY κ_def,0 with same-sign mask:
        κ_def,0(t) = Σ_d w_d * |ΔP_0(t-d)|/|ΔP_0(t)|, if sign(ΔP_0(t-d)) == sign(ΔP_0(t))
  - Keep mean/max over steps [100, 4999]

Outputs:
  - kappa_def_vs_inbound_mean.pdf / .png
  - kappa_def_vs_inbound_max.pdf  / .png
  - kappa_def_vs_inbound_{Spike|Uniform|Bimodal}.csv
  - kappa_def_vs_inbound_all.csv
"""

import os, glob, time, shutil, subprocess
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Sweep setup ----------------

LAMB_MIN, LAMB_MAX, LAMB_STEP = 4.0, 16.0, 0.5
ALPHA_MIN, ALPHA_MAX          = 0.45, 0.95

EQUIL_P    = 1.0
START_STEP = 100
END_STEP   = 4999

DISTS = {
    "Spike"  : [0, 0, 0, 0, 1],
    "Uniform": [0.2, 0.2, 0.2, 0.2, 0.2],
    "Bimodal": [0.4, 0, 0.2, 0, 0.4],
}
TITLE_FONTSIZE  = 16
LABEL_FONTSIZE  = 14
TICK_FONTSIZE   = 12
LEGEND_FONTSIZE = 11
plt.rcParams.update({
    "axes.titlesize":  TITLE_FONTSIZE,
    "axes.labelsize":  LABEL_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
})

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

def build_self_consistent_inbound_matrix(T, alpha):
    """3x3 base_demand_matrix satisfying shard totals & α inbound to shard 0."""
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

def set_lambda_j0(cfg, lamb):
    """Set λ_{10} and λ_{20} to 'lamb' (other entries unchanged)."""
    lam = cfg['demand']['lambda_matrix']
    lam[1][0] = float(lamb)
    lam[2][0] = float(lamb)

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
    """
    Parse latest enhanced_simulation_analysis_*.log
    Supports:
      step,fee0,fee1,fee2,load0,load1,load2
      or fallback: step,fee,load (single shard)
    Returns: dict {0: df0, 1: df1, 2: df2}
    """
    files = glob.glob('enhanced_simulation_analysis_*.log')
    if not files:
        return None
    latest = max(files, key=os.path.getctime)
    data = {0: [], 1: [], 2: []}
    with open(latest, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            parts = line.split(',')
            try:
                if len(parts) >= 7:
                    step = int(parts[0])
                    fees = [float(parts[1]), float(parts[2]), float(parts[3])]
                    loads = [float(parts[4]), float(parts[5]), float(parts[6])]
                    for sid in (0, 1, 2):
                        data[sid].append({'Step': step, 'Fee': fees[sid], 'Load': loads[sid]})
                elif len(parts) >= 3:
                    step = int(parts[0]); fee = float(parts[1]); load = float(parts[2])
                    data[0].append({'Step': step, 'Fee': fee, 'Load': load})
            except:
                continue

    out = {}
    for sid in (0, 1, 2):
        if data[sid]:
            out[sid] = pd.DataFrame(data[sid])
    return out if out else None

# --------------- κ_def(t) calculation (same-sign) --

def kappa_def_timeseries(delta_p0, weights, start_idx, end_idx):
    """
    κ_def,0(t) = Σ_d w_d * |ΔP0(t-d)| / |ΔP0(t)|   if sign(ΔP0(t-d)) == sign(ΔP0(t)).
    """
    dmax = len(weights)
    end_idx = min(end_idx, len(delta_p0) - 1)
    if start_idx + dmax > end_idx:
        return []

    ks = []
    eps = 1e-12
    for t in range(start_idx + dmax, end_idx + 1):
        cur = delta_p0[t]
        if not np.isfinite(cur) or abs(cur) < eps:
            continue
        s_cur = np.sign(cur)
        k = 0.0
        for d in range(1, dmax + 1):
            prev = delta_p0[t - d]
            if not np.isfinite(prev) or prev == 0.0:
                continue
            if np.sign(prev) == s_cur:  # same-sign enhancement
                k += weights[d - 1] * (abs(prev) / abs(cur))
        ks.append(k)
    return ks

# --------------- Mapping α from λ ------------------

def alpha_from_lambda(lamb):
    """Map λ ∈ [4,16] linearly to α ∈ [0.45,0.95]."""
    return ALPHA_MIN + (lamb - LAMB_MIN) / (LAMB_MAX - LAMB_MIN) * (ALPHA_MAX - ALPHA_MIN)

# --------------- Sweep for one distribution --------

def sweep_for_distribution(dist_name, weights, T):
    lamb_values  = np.arange(LAMB_MIN, LAMB_MAX + 1e-9, LAMB_STEP)
    alpha_values = np.array([alpha_from_lambda(l) for l in lamb_values])
    labels_pairs = [f"({l:.1f}, {a:.3f})" for l, a in zip(lamb_values, alpha_values)]

    means, maxes = [], []

    for lamb, alpha, lab in zip(lamb_values, alpha_values, labels_pairs):
        print(f"\n[{dist_name}] (λ, α) = {lab}")
        cfg = load_cfg()

        set_delay_weights(cfg, weights)
        set_lambda_j0(cfg, lamb)
        M = build_self_consistent_inbound_matrix(T, alpha)
        set_base_matrix(cfg, M)
        save_cfg(cfg)

        if not run_sim():
            means.append(np.nan); maxes.append(np.nan); continue

        dfs = parse_latest_log()
        if not dfs or 0 not in dfs or dfs[0].empty:
            print("  ⚠️ missing shard0 data")
            means.append(np.nan); maxes.append(np.nan); continue

        df0 = dfs[0]
        if not (df0['Step'] >= START_STEP).any() or not (df0['Step'] <= END_STEP).any():
            print("  ⚠️ window not found")
            means.append(np.nan); maxes.append(np.nan); continue

        start_idx = df0[df0['Step'] >= START_STEP].index[0]
        end_idx   = df0[df0['Step'] <= END_STEP].index[-1]

        dP0 = (df0['Fee'].values - EQUIL_P)
        ks  = kappa_def_timeseries(dP0, weights, start_idx, end_idx)

        if ks:
            mean_k = float(np.mean(ks)); max_k = float(np.max(ks))
            print(f"  mean κ_def = {mean_k:.4f},  max κ_def = {max_k:.4f}")
        else:
            mean_k = np.nan; max_k = np.nan
            print("  κ_def series empty; mean/max = NaN")

        means.append(mean_k)
        maxes.append(max_k)

    # save per-distribution CSV
    out_rows = [
        {'distribution': dist_name,
         'lambda_j0': float(l),
         'alpha_j0_in': float(a),
         'mean_kappa_def': (float(m) if m == m else None),
         'max_kappa_def' : (float(x) if x == x else None)}
        for l, a, m, x in zip(lamb_values, alpha_values, means, maxes)
    ]
    pd.DataFrame(out_rows).to_csv(f'kappa_def_vs_inbound_{dist_name}.csv', index=False)
    print(f"💾 saved kappa_def_vs_inbound_{dist_name}.csv")

    return lamb_values, alpha_values, labels_pairs, np.array(means), np.array(maxes)

# --------------- Main -----------------------------

def main():
    backup_cfg()
    try:
        cfg0 = load_cfg()
        T = target_T(cfg0)

        results = {}
        for name, w in DISTS.items():
            lambs, alphas, labels_pairs, means, maxes = sweep_for_distribution(name, w, T)
            results[name] = {
                'lamb': lambs, 'alpha': alphas, 'labels_pairs': labels_pairs,
                'mean': means, 'max': maxes
            }

        # save combined CSV
        combined = []
        for name, r in results.items():
            for l, a, m, x in zip(r['lamb'], r['alpha'], r['mean'], r['max']):
                combined.append({
                    'distribution': name,
                    'lambda_j0': float(l),
                    'alpha_j0_in': float(a),
                    'mean_kappa_def': (float(m) if m == m else None),
                    'max_kappa_def' : (float(x) if x == x else None),
                })
        pd.DataFrame(combined).to_csv('kappa_def_vs_inbound_all.csv', index=False)
        print("💾 saved kappa_def_vs_inbound_all.csv")

        # common x grid and 5 evenly-spaced ticks
        x = np.arange(len(results['Spike']['labels_pairs']))
        xtick_idx = np.linspace(0, len(x) - 1, 5).astype(int)
        xtick_labels = [results['Spike']['labels_pairs'][i] for i in xtick_idx]

        style = {'Spike': 'o', 'Uniform': 's', 'Bimodal': 'D'}

        # 1) Mean κ_def figure
        plt.figure(figsize=(6.2, 3.4))
        for name in ['Spike', 'Uniform', 'Bimodal']:
            r = results[name]
            plt.plot(x, r['mean'], '-', marker=style[name], markersize=6,
                     linewidth=1.8, label=name)
        plt.xticks(xtick_idx, xtick_labels)
        plt.xlabel('($λ_{j0}$, $\sumα_{j0,in}$) pairs')
        plt.ylabel('Mean κ')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('kappa_def_vs_inbound_mean.pdf')
        plt.savefig('kappa_def_vs_inbound_mean.png', dpi=240)
        print("📈 saved kappa_def_vs_inbound_mean.pdf / .png")

        # 2) Max κ_def figure
        plt.figure(figsize=(6.2, 3.4))
        for name in ['Spike', 'Uniform', 'Bimodal']:
            r = results[name]
            plt.plot(x, r['max'], '--', marker=style[name], markersize=6,
                     linewidth=1.7, label=name)
        plt.xticks(xtick_idx, xtick_labels)
        plt.xlabel('($λ_{j0}$, $\sumα_{j0,in}$) pairs')
        plt.ylabel('Max κ')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')  # second figure legend in upper-left
        plt.tight_layout()
        plt.savefig('kappa_def_vs_inbound_max.pdf')
        plt.savefig('kappa_def_vs_inbound_max.png', dpi=240)
        print("📈 saved kappa_def_vs_inbound_max.pdf / .png")

    finally:
        restore_cfg()

if __name__ == "__main__":
    main()
