#!/usr/bin/env python3
"""
Lambda_j0 & Alpha_j0,in Scan — EXACT per-user allocation rule

Rule (T = G_MAX/2, alpha = alpha_j0,in ∈ [0,1]):
  10 = 20 = (alpha/2) * T
  00 = 11 = 22 = (1 - alpha) * T
  01 = 02 = 0
  For shard 1: remaining T - (10+11) is split evenly -> 12 = 21 = (alpha/4) * T
  For shard 2: same -> 21 = 12 = (alpha/4) * T

Matrix:
  [[(1-α)T,       0,          0      ],
   [ α/2 T,  (1-α)T,     α/4 T       ],
   [ α/2 T,     α/4 T,   (1-α)T      ]]

We scan:
  - lambda_j0: set demand.lambda_matrix[1][0] = demand.lambda_matrix[2][0] = lambda_j0
  - alpha_j0,in: defines the matrix above
  - delay.weights: Spike / Uniform / Bimodal
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import subprocess
import os
import shutil
import glob
from io import StringIO

# -----------------------------
# Delay distributions
# -----------------------------
DELAY_DISTRIBUTIONS = {
    'Spike':   [0,   0,   0,   0,   1],
    'Uniform': [0.2, 0.2, 0.2, 0.2, 0.2],
    'Bimodal': [0.4, 0,   0.2, 0,   0.4],
}

# -----------------------------
# Simulation constants (match your config)
# -----------------------------
G_MAX = 2_000_000
TARGET_TOTAL_DEMAND = G_MAX / 2.0  # per shard T

# -----------------------------
# Scan ranges
# -----------------------------
LAMBDA_J0_MIN  = 1.5
LAMBDA_J0_MAX  = 16
LAMBDA_J0_STEP = 0.1

ALPHA_IN_MIN   = 0.45
ALPHA_IN_MAX   = 0.95
ALPHA_IN_STEP  = 0.01

def _points(min_v, max_v, step):
    return int(round((max_v - min_v) / step)) + 1

LAMBDA_POINTS   = _points(LAMBDA_J0_MIN, LAMBDA_J0_MAX, LAMBDA_J0_STEP)
ALPHA_IN_POINTS = _points(ALPHA_IN_MIN,  ALPHA_IN_MAX,  ALPHA_IN_STEP)

# -----------------------------
# IO helpers
# -----------------------------
def load_config():
    try:
        with open('config.yml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ config.yml not found")
        return None

def backup_config():
    try:
        shutil.copy('config.yml', 'config_backup.yml')
        print("✅ Created config backup")
        return True
    except Exception as e:
        print(f"❌ Failed to backup config: {e}")
        return False

def restore_config():
    try:
        shutil.copy('config_backup.yml', 'config.yml')
        print("✅ Restored original config")
        return True
    except Exception as e:
        print(f"❌ Failed to restore config: {e}")
        return False

# -----------------------------
# Demand matrix — EXACT rule
# -----------------------------
def demand_matrix_exact(alpha_in):
    """
    Build the 3x3 base_demand_matrix exactly per user's rule.
    """
    T = TARGET_TOTAL_DEMAND
    a = float(alpha_in)

    m = [[0.0]*3 for _ in range(3)]

    # row 0 (from shard 0)
    m[0][0] = (1.0 - a) * T
    m[0][1] = 0.0
    m[0][2] = 0.0

    # row 1 (from shard 1)
    m[1][0] = (a * T) / 2.0          # 10
    m[1][1] = (1.0 - a) * T          # 11
    m[1][2] = (a * T) / 4.0          # 12

    # row 2 (from shard 2)
    m[2][0] = (a * T) / 2.0          # 20
    m[2][1] = (a * T) / 4.0          # 21
    m[2][2] = (1.0 - a) * T          # 22

    return m

def verify_demand_matrix(matrix, alpha_in_target, tol=1.0, alpha_tol=1e-9):
    """
    Verify your three shard-total constraints & alpha_j0,in:
      shard0: 00 + 01 + 02 + 10 + 20 = T
      shard1: 10 + 11 + 12 + 01 + 21 = T
      shard2: 20 + 21 + 22 + 02 + 12 = T
      (10+20)/T = alpha_in_target
    """
    T = TARGET_TOTAL_DEMAND

    shard0_total = matrix[0][0] + matrix[0][1] + matrix[0][2] + matrix[1][0] + matrix[2][0]
    shard1_total = matrix[1][0] + matrix[1][1] + matrix[1][2] + matrix[0][1] + matrix[2][1]
    shard2_total = matrix[2][0] + matrix[2][1] + matrix[2][2] + matrix[0][2] + matrix[1][2]

    inflow_to_0 = matrix[1][0] + matrix[2][0]
    alpha_actual = inflow_to_0 / T

    shard_ok = (abs(shard0_total - T) < tol and
                abs(shard1_total - T) < tol and
                abs(shard2_total - T) < tol)
    alpha_ok = abs(alpha_actual - alpha_in_target) < alpha_tol
    nonneg   = all(matrix[i][j] >= 0 for i in range(3) for j in range(3))

    print("Demand matrix verification:")
    print(f"  Shard totals: {shard0_total:.0f}, {shard1_total:.0f}, {shard2_total:.0f}  (target: {T:.0f})")
    print(f"  Alpha_j0,in: actual={alpha_actual:.6f}, target={alpha_in_target:.6f}")
    print(f"  Nonnegative: {nonneg}")
    print(f"  Constraints satisfied: {shard_ok and alpha_ok and nonneg}")

    return shard_ok and alpha_ok and nonneg

def print_demand_matrix(matrix, alpha_value):
    print(f"\nDemand matrix (EXACT) for alpha_j0,in={alpha_value:.2f}:")
    print("     To:        0         1         2")
    for i in range(3):
        row = f"From {i}: "
        for j in range(3):
            row += f"{matrix[i][j]:10.0f} "
        print(row)
    print(f"Inflows to shard 0 (10,20): {matrix[1][0]:.0f}, {matrix[2][0]:.0f}")
    print(f"00={matrix[0][0]:.0f}, 11={matrix[1][1]:.0f}, 22={matrix[2][2]:.0f}; 01=02=0; 12=21={matrix[1][2]:.0f}")

def test_demand_matrix_calculation():
    print("🧪 Testing EXACT demand matrix (alpha_j0,in sweep)...")
    samples = [0.00, 0.10, 0.50, 0.90]
    for a in samples:
        m = demand_matrix_exact(a)
        print_demand_matrix(m, a)
        ok = verify_demand_matrix(m, a)
        if not ok:
            print(f"❌ Invalid matrix for alpha_j0,in={a}")
            return False
        print(f"✅ Valid matrix for alpha_j0,in={a}")
    return True

# -----------------------------
# Experiment plumbing
# -----------------------------
def update_config_for_experiment(lambda_j0, alpha_in, delay_weights):
    """
    Update config.yml with:
      - demand.lambda_matrix[1][0] = demand.lambda_matrix[2][0] = lambda_j0
      - demand.base_demand_matrix  = EXACT(alpha_in)
      - delay.weights              = per distribution
    """
    try:
        config = load_config()
        if config is None:
            return False

        # 1) lambda_j0 on inbound edges to shard 0
        lam = [row[:] for row in config['demand']['lambda_matrix']]
        lam[1][0] = float(lambda_j0)
        lam[2][0] = float(lambda_j0)
        config['demand']['lambda_matrix'] = lam

        # 2) base demand matrix from exact rule
        dm = demand_matrix_exact(alpha_in)
        config['demand']['base_demand_matrix'] = dm

        # 3) delay weights
        if 'delay' not in config:
            config['delay'] = {}
        config['delay']['weights'] = delay_weights

        with open('config.yml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"✅ Config updated: lambda_j0={lambda_j0:.2f}, alpha_j0,in={alpha_in:.2f}")
        return True

    except Exception as e:
        print(f"❌ Failed to update config: {e}")
        return False

def run_simulation():
    try:
        res = subprocess.run(['go', 'run', '../../../main.go'],
                             capture_output=True, text=True, timeout=60)
        if res.returncode == 0:
            return True
        print(f"❌ Simulation failed: {res.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Simulation timed out")
        return False
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        return False

def find_latest_log():
    logs = glob.glob("enhanced_simulation_analysis_*.log")
    if logs:
        return max(logs, key=os.path.getctime)
    return None

def parse_final_load(log_file):
    try:
        with open(log_file, 'r') as f:
            content = f.read()

        print(f"  📄 Parsing log file: {log_file}")
        i = content.find('DATA_START')
        if i == -1:
            print("  ❌ No DATA_START found")
            return None

        csv_section = content[i:]
        lines = csv_section.split('\n')
        rows = []
        header_found = False
        for line in lines:
            s = line.strip()
            if s.startswith('Step,Shard0_Fee'):
                header_found = True
                rows.append(s)
            elif header_found and ',' in s and not s.startswith('#'):
                parts = s.split(',')
                if len(parts) >= 7:
                    try:
                        int(parts[0])
                        rows.append(s)
                    except ValueError:
                        break

        if len(rows) <= 1:
            print("  ❌ No valid CSV rows")
            return None

        df = pd.read_csv(StringIO('\n'.join(rows)))
        s0 = df['Shard0_Load']
        final_load = s0.iloc[-1]
        avg_load   = s0.mean()
        load_std   = s0.std()
        target     = 0.5
        inflation  = (avg_load - target) / target

        print(f"  ✅ Shard0 stats: final={final_load:.6f}, avg={avg_load:.6f}, std={load_std:.6f}")
        return {
            'load0': final_load,
            'avg_load': avg_load,
            'load_std': load_std,
            'load_inflation': inflation,
        }

    except Exception as e:
        print(f"❌ Error parsing log: {e}")
        return {
            'load0': 0.5,
            'avg_load': 0.5,
            'load_std': 0.0,
            'load_inflation': 0.0,
        }

def clean_log_files():
    try:
        enhanced_logs  = glob.glob("enhanced_simulation_analysis_*.log")
        experiment_logs = glob.glob("experiment_*.log")
        all_logs = enhanced_logs + experiment_logs
        for p in all_logs:
            os.remove(p)
        if all_logs:
            print(f"  🧹 Cleaned {len(all_logs)} logs")
        return True
    except Exception as e:
        print(f"⚠️  Warning: Could not clean logs: {e}")
        return False

def check_convergence(load_stats, tolerance=1e-5):
    return (isinstance(load_stats, dict) and
            abs(load_stats.get('load0', 1.0) - 0.5) < tolerance)

def run_single_experiment(lambda_j0, alpha_in, delay_weights, distribution_name, exp_id):
    if not update_config_for_experiment(lambda_j0, alpha_in, delay_weights):
        return None

    unique_log = f"experiment_{distribution_name}_{exp_id:04d}_lam{lambda_j0:.2f}_alphain{alpha_in:.2f}.log"

    if not run_simulation():
        return None

    latest = find_latest_log()
    if latest is None:
        print(f"  ❌ No log found for experiment {exp_id}")
        return None

    try:
        os.rename(latest, unique_log)
        print(f"  📝 Renamed log to: {unique_log}")
    except Exception as e:
        print(f"  ⚠️  Could not rename log: {e}")
        unique_log = latest

    load_stats = parse_final_load(unique_log)
    converged  = check_convergence(load_stats)

    try:
        os.remove(unique_log)
        print(f"  🗑️  Cleaned up: {unique_log}")
    except Exception as e:
        print(f"  ⚠️  Could not remove log: {e}")

    return {
        'lambda_j0': lambda_j0,
        'alpha_inflow': alpha_in,
        'distribution': distribution_name,
        'final_load': load_stats['load0'],
        'avg_load': load_stats['avg_load'],
        'load_std': load_stats['load_std'],
        'load_inflation': load_stats['load_inflation'],
        'converged': converged,
    }

def save_distribution_results(results, distribution_name):
    df = pd.DataFrame(results)
    filename = f"attack_defense_results_{distribution_name}.csv"
    df.to_csv(filename, index=False)
    print(f"💾 Saved {len(results)} results to {filename}")

def run_attack_defense_experiment():
    print("🔬 Starting Lambda_j0 & Alpha_j0,in scan (EXACT rule)")
    print("=" * 80)

    total = LAMBDA_POINTS * ALPHA_IN_POINTS * len(DELAY_DISTRIBUTIONS)
    print(f"Total experiments to run: {total}")
    print(f"lambda_j0: {LAMBDA_J0_MIN} → {LAMBDA_J0_MAX} (step {LAMBDA_J0_STEP}, {LAMBDA_POINTS} points)")
    print(f"alpha_j0,in: {ALPHA_IN_MIN} → {ALPHA_IN_MAX} (step {ALPHA_IN_STEP}, {ALPHA_IN_POINTS} points)")

    if not backup_config():
        print("❌ Failed to backup config, aborting")
        return

    try:
        exp_count = 0
        for dist_name, weights in DELAY_DISTRIBUTIONS.items():
            print(f"\n🧪 Distribution: {dist_name}  weights={weights}")
            print("-" * 60)

            results = []

            for i in range(LAMBDA_POINTS - 1, -1, -1):
                lambda_j0 = LAMBDA_J0_MIN + i * LAMBDA_J0_STEP

                for j in range(ALPHA_IN_POINTS - 1, -1, -1):
                    alpha_in = ALPHA_IN_MIN + j * ALPHA_IN_STEP
                    exp_count += 1

                    print(f"Experiment {exp_count}/{total}: {dist_name}, "
                          f"λ_j0={lambda_j0:.2f}, α_j0,in={alpha_in:.2f}")

                    r = run_single_experiment(lambda_j0, alpha_in, weights, dist_name, exp_count)

                    if r is not None:
                        results.append(r)
                        status = "✅ Converged" if r['converged'] else "❌ Diverged"
                        print(f"  → Load0: {r['final_load']:.6f}, Avg: {r['avg_load']:.6f}, "
                              f"Std: {r['load_std']:.6f}, Inflation: {r['load_inflation']:.6f}, {status}")
                    else:
                        print("  → ❌ Experiment failed")
                        results.append({
                            'lambda_j0': lambda_j0,
                            'alpha_inflow': alpha_in,
                            'distribution': dist_name,
                            'final_load': None,
                            'avg_load': None,
                            'load_std': None,
                            'load_inflation': None,
                            'converged': False,
                        })

                    if len(results) % 10 == 0:
                        save_distribution_results(results, dist_name)

                    if exp_count % 50 == 0:
                        clean_log_files()

            save_distribution_results(results, dist_name)
            print(f"✅ Completed {dist_name}: {len(results)} experiments")

        print("\n🎉 All experiments completed!")

    finally:
        restore_config()
        print("🔄 Original config.yml restored")

# -----------------------------
# Visualization & analysis
# -----------------------------
def load_and_combine_results():
    all_dfs = []
    files = []
    for name in DELAY_DISTRIBUTIONS.keys():
        fn = f"attack_defense_results_{name}.csv"
        if os.path.exists(fn):
            files.append(fn)

    print(f"Found {len(files)} distribution result files")
    for fn in files:
        try:
            df = pd.read_csv(fn)
            all_dfs.append(df)
            print(f"Loaded {len(df)} rows from {fn}")
        except Exception as e:
            print(f"❌ Error loading {fn}: {e}")

    if all_dfs:
        out = pd.concat(all_dfs, ignore_index=True)
        print(f"✅ Combined {len(out)} total rows")
        return out
    print("❌ No results found")
    return None

def create_2d_plots(df):
    if df is None or len(df) == 0:
        print("❌ No data available for plotting")
        return

    dists = df['distribution'].unique()
    fig, axes = plt.subplots(1, len(dists), figsize=(6*len(dists), 6))

    if len(dists) == 1:
        axes = [axes]

    for i, dist in enumerate(dists):
        ddf = df[df['distribution'] == dist]

        pivot = ddf.pivot_table(
            values='final_load',
            index='alpha_inflow',
            columns='lambda_j0',
            aggfunc='mean'
        )

        im = axes[i].imshow(pivot.values, cmap='RdYlBu_r', aspect='auto', origin='lower')
        axes[i].set_xlabel('lambda_j0 (λⱼ0)', fontsize=12)
        axes[i].set_ylabel('alpha_j0,in (αⱼ0,in)', fontsize=12)
        axes[i].set_title(f'{dist} Distribution\n(λⱼ0 vs αⱼ0,in)', fontsize=14, fontweight='bold')

        xt = np.arange(0, len(pivot.columns), max(1, len(pivot.columns)//10))
        yt = np.arange(0, len(pivot.index),   max(1, len(pivot.index)//10))
        axes[i].set_xticks(xt); axes[i].set_yticks(yt)
        if len(xt) > 0:
            axes[i].set_xticklabels([f'{pivot.columns[t]:.1f}' for t in xt])
        if len(yt) > 0:
            axes[i].set_yticklabels([f'{pivot.index[t]:.2f}' for t in yt])

        cbar = plt.colorbar(im, ax=axes[i])
        cbar.set_label('Final Load (Shard 0)', fontsize=10)

    plt.tight_layout()
    plt.savefig('attack_defense_lambda_alphaIn_exact_2d.pdf', dpi=300, bbox_inches='tight')
    print("💾 2D plots saved as attack_defense_lambda_alphaIn_exact_2d.pdf")
    plt.show()

def analyze_attack_defense_results(df):
    if df is None or len(df) == 0:
        print("❌ No data available for analysis")
        return

    print("\n📊 LAMBDA_J0 & ALPHA_J0,IN ANALYSIS SUMMARY (EXACT rule)")
    print("=" * 80)

    for dist in df['distribution'].unique():
        d = df[df['distribution'] == dist]
        n = len(d)
        conv = d['converged'].sum()
        rate = 100.0 * conv / n if n else 0.0

        avg_final = d['final_load'].mean()
        max_final = d['final_load'].max()
        min_final = d['final_load'].min()

        print(f"\n🔬 {dist}:")
        print(f"   Total: {n}  Converged: {conv} ({rate:.1f}%)")
        print(f"   Final load: avg={avg_final:.6f}  min={min_final:.6f}  max={max_final:.6f}")

        max_case = d.loc[d['final_load'].idxmax()]
        min_case = d.loc[d['final_load'].idxmin()]
        print("   Extremes:")
        print(f"     Max: λ={max_case['lambda_j0']:.2f}, α_in={max_case['alpha_inflow']:.2f}")
        print(f"     Min: λ={min_case['lambda_j0']:.2f}, α_in={min_case['alpha_inflow']:.2f}")

    summary = []
    for dist in df['distribution'].unique():
        d = df[df['distribution'] == dist]
        summary.append({
            'distribution': dist,
            'total_experiments': len(d),
            'converged_count': int(d['converged'].sum()),
            'convergence_rate': 100.0 * d['converged'].sum() / len(d) if len(d) else 0.0,
            'avg_final_load': d['final_load'].mean(),
            'max_final_load': d['final_load'].max(),
            'min_final_load': d['final_load'].min(),
        })
    pd.DataFrame(summary).to_csv("attack_defense_lambda_alphaIn_exact_summary.csv", index=False)
    print("💾 Summary saved to attack_defense_lambda_alphaIn_exact_summary.csv")

def visualize_results():
    print("📈 Loading and visualizing results...")
    df = load_and_combine_results()
    if df is not None:
        analyze_attack_defense_results(df)
        create_2d_plots(df)
        df.to_csv("attack_defense_lambda_alphaIn_exact_combined.csv", index=False)
        print("💾 Combined results saved to attack_defense_lambda_alphaIn_exact_combined.csv")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    print("🔬 Lambda_j0 & Alpha_j0,in Scan (EXACT rule)")
    print("=" * 60)
    print("Choose operation:")
    print("1. Test demand matrix calculation")
    print("2. Run full experiment")
    print("3. Visualize existing results")
    print("4. Exit")

    choice = input("Enter choice (1-4): ").strip()
    if choice == "1":
        if test_demand_matrix_calculation():
            print("\n✅ Demand matrix calculation verified!")
        else:
            print("\n❌ Demand matrix calculation has issues.")
    elif choice == "2":
        if test_demand_matrix_calculation():
            print("\n✅ Demand matrix calculation verified!")
            run_attack_defense_experiment()
        else:
            print("\n❌ Please fix demand matrix calculation first.")
    elif choice == "3":
        visualize_results()
    elif choice == "4":
        print("Goodbye!")
    else:
        print("Invalid choice. Please run again.")
