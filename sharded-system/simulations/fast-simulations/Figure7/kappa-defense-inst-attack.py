#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scan (epsilon_{0j}, alpha_{0j,out}) and measure DEFENSE kappa (shard 0) under three latency distributions.
- Modify epsilon_matrix[0][1], epsilon_matrix[0][2] = ε   (attack-side destination elasticity)
- Build self-consistent base_demand_matrix emphasizing OUTBOUND from shard 0:
      T = g_max(shard0) * L_target
      01 = 02 = α * T / 2
      10 = 20 = 0
      00 = 11 = 22 = (1 - α) * T
      12 = 21 = α * T / 4
- Compute κ_def,0(t) = Σ_d w_d * |ΔP_0(t-d)| / |ΔP_0(t)| * 1(ΔP_0(t-d)·ΔP_0(t) > 0)
- Keep only mean/max over steps [100, 4999]
- Plot two separate figures (mean / max) with 5 x-ticks; legends only show distribution names.
- Second figure legend loc='upper left'.
"""

import os, glob, time, shutil, subprocess
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- sweep ranges ----------
EPS_MIN, EPS_MAX, EPS_STEP = 4.0, 16.0, 0.5      # ε_0j
ALPHA_MIN, ALPHA_MAX       = 0.45, 0.95          # α_{0j,out} mapped linearly to ε

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


# ---------- config IO ----------
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

# ---------- helpers ----------
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

def build_matrix_outbound(T, alpha):
    """Self-consistent base_demand_matrix focusing on OUTBOUND from shard 0."""
    alpha = max(0.0, min(1.0, float(alpha)))
    g00 = (1 - alpha) * T
    g01 = g02 = 0.5 * alpha * T
    g10 = g20 = 0.0
    g11 = g22 = g00
    g12 = g21 = 0.25 * alpha * T
    return [
        [g00, g01, g02],
        [g10, g11, g12],
        [g20, g21, g22],
    ]

def set_eps_0j(cfg, eps):
    epsM = cfg['demand']['epsilon_matrix']
    epsM[0][1] = float(eps)
    epsM[0][2] = float(eps)

def set_delay_weights(cfg, weights):
    cfg['delay']['weights'] = [float(w) for w in weights]

def set_base_matrix(cfg, M):
    cfg['demand']['base_demand_matrix'] = [[float(x) for x in row] for row in M]

def alpha_from_eps(eps):
    """Map ε ∈ [4,16] to α ∈ [0.45,0.95] linearly."""
    return ALPHA_MIN + (eps - EPS_MIN) / (EPS_MAX - EPS_MIN) * (ALPHA_MAX - ALPHA_MIN)

# ---------- sim & parse ----------
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
                    fees  = [float(parts[1]), float(parts[2]), float(parts[3])]
                    loads = [float(parts[4]), float(parts[5]), float(parts[6])]
                    for sid in (0,1,2):
                        data[sid].append({'Step': step, 'Fee': fees[sid], 'Load': loads[sid]})
                elif len(parts) >= 3:
                    step = int(parts[0]); fee = float(parts[1]); load = float(parts[2])
                    data[0].append({'Step': step, 'Fee': fee, 'Load': load})
            except:
                continue
    dfs = {}
    for sid in data:
        if data[sid]:
            dfs[sid] = pd.DataFrame(data[sid])
    return dfs if dfs else None

# ---------- κ_def (same-sign mask, shard 0 only) ----------
def kappa_def_timeseries(delta_p0, weights, start_idx, end_idx):
    """
    κ_def,0(t) = Σ_d w_d * |ΔP0(t-d)| / |ΔP0(t)|   if sign(ΔP0(t-d)) == sign(ΔP0(t)).
    """
    dmax = len(weights)
    end_idx = min(end_idx, len(delta_p0)-1)
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
        for d in range(1, dmax+1):
            prev = delta_p0[t-d]
            if not np.isfinite(prev) or prev == 0.0:
                continue
            if np.sign(prev) == s_cur:  # same-sign enhancement
                k += weights[d-1] * (abs(prev)/abs(cur))
        ks.append(k)
    return ks

# ---------- sweep for one distribution ----------
def sweep_distribution(dist_name, weights, T):
    eps_values   = np.arange(EPS_MIN, EPS_MAX + 1e-9, EPS_STEP)
    alpha_values = np.array([alpha_from_eps(e) for e in eps_values])
    labels_pairs = [f"({e:.1f}, {a:.3f})" for e,a in zip(eps_values, alpha_values)]

    means, maxes = [], []

    for eps, alpha, lab in zip(eps_values, alpha_values, labels_pairs):
        print(f"\n[{dist_name}] {lab}")
        cfg = load_cfg()
        set_delay_weights(cfg, weights)
        set_eps_0j(cfg, eps)
        M = build_matrix_outbound(T, alpha)
        set_base_matrix(cfg, M)
        save_cfg(cfg)

        if not run_sim():
            means.append(np.nan); maxes.append(np.nan); continue

        df_dict = parse_latest_log()
        if df_dict is None or (0 not in df_dict) or df_dict[0].empty:
            print("  ⚠️ missing shard0 series"); means.append(np.nan); maxes.append(np.nan); continue

        df0 = df_dict[0]
        if not (df0['Step']>=START_STEP).any() or not (df0['Step']<=END_STEP).any():
            print("  ⚠️ window not found"); means.append(np.nan); maxes.append(np.nan); continue

        start_idx = df0[df0['Step']>=START_STEP].index[0]
        end_idx   = df0[df0['Step']<=END_STEP].index[-1]
        dp0 = (df0['Fee'].values - EQUIL_P)

        ks = kappa_def_timeseries(dp0, weights, start_idx, end_idx)
        if ks:
            mean_k = float(np.mean(ks)); max_k = float(np.max(ks))
            print(f"  mean κ_def = {mean_k:.4f},  max κ_def = {max_k:.4f}")
        else:
            mean_k = np.nan; max_k = np.nan
            print("  κ_def series empty; mean/max = NaN")

        means.append(mean_k); maxes.append(max_k)

    out = pd.DataFrame({
        'distribution': dist_name,
        'epsilon_0j': eps_values,
        'alpha_0j_out': alpha_values,
        'mean_kappa_def': means,
        'max_kappa_def' : maxes,
    })
    out.to_csv(f'kappa_def_vs_outbound_{dist_name}.csv', index=False)
    print(f"💾 saved kappa_def_vs_outbound_{dist_name}.csv")
    return eps_values, alpha_values, labels_pairs, np.array(means), np.array(maxes)

# ---------- main ----------
def main():
    backup_cfg()
    try:
        cfg0 = load_cfg()
        T = target_T(cfg0)

        results = {}
        for name, w in DISTS.items():
            eps, alp, labels, mean_arr, max_arr = sweep_distribution(name, w, T)
            results[name] = {'eps': eps, 'alpha': alp, 'labels': labels,
                             'mean': mean_arr, 'max': max_arr}

        # combined CSV
        comb = []
        for name, r in results.items():
            for e, a, m, x in zip(r['eps'], r['alpha'], r['mean'], r['max']):
                comb.append({'distribution': name, 'epsilon_0j': float(e),
                             'alpha_0j_out': float(a),
                             'mean_kappa_def': (float(m) if m==m else None),
                             'max_kappa_def' : (float(x) if x==x else None)})
        pd.DataFrame(comb).to_csv('kappa_def_vs_outbound_all.csv', index=False)
        print("💾 saved kappa_def_vs_outbound_all.csv")

        # x grid & 5 ticks
        x = np.arange(len(results['Spike']['labels']))
        tick_idx = np.linspace(0, len(x)-1, 5).astype(int)
        tick_labels = [results['Spike']['labels'][i] for i in tick_idx]

        style = {'Spike': {'m':'o'}, 'Uniform': {'m':'s'}, 'Bimodal': {'m':'D'}}

        # ---- figure: mean κ_def ----
        plt.figure(figsize=(6.2, 3.4))
        for name in ['Spike','Uniform','Bimodal']:
            r = results[name]
            plt.plot(x, r['mean'], '-', marker=style[name]['m'], markersize=6,
                     linewidth=1.8, label=name)
        plt.xticks(tick_idx, tick_labels)
        plt.xlabel('($ε_{0j}$, $\sumα_{0j,out}$) pairs')
        plt.ylabel('Mean κ')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('kappa_def_vs_outbound_mean.pdf')
        plt.savefig('kappa_def_vs_outbound_mean.png', dpi=240)
        print("📈 saved kappa_def_vs_outbound_mean.pdf / .png")

        # ---- figure: max κ_def ----
        plt.figure(figsize=(6.2, 3.4))
        for name in ['Spike','Uniform','Bimodal']:
            r = results[name]
            plt.plot(x, r['max'], '--', marker=style[name]['m'], markersize=6,
                     linewidth=1.7, label=name)
        plt.xticks(tick_idx, tick_labels)
        plt.xlabel('($ε_{0j}$, $\sumα_{0j,out}$) pairs')
        plt.ylabel('Max κ')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')  # second figure legend in the upper-left
        plt.tight_layout()
        plt.savefig('kappa_def_vs_outbound_max.pdf')
        plt.savefig('kappa_def_vs_outbound_max.png', dpi=240)
        print("📈 saved kappa_def_vs_outbound_max.pdf / .png")

    finally:
        restore_cfg()

if __name__ == '__main__':
    main()
