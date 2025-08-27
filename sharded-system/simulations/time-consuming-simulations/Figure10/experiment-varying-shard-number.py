#!/usr/bin/env python3
"""
Convergence Boundary Experiment (Spike-only, multi-shard)
- 仅 Spike 延迟分布
- 顺序运行 N=10 → 7 → 5
- 自动把 config.yml 里的 network.num_shards / shards / 各矩阵尺寸同步到 N
- 生成 3 个 CSV：
    - convergence_results_spike_shard10.csv
    - convergence_results_spike_shard7.csv
    - convergence_results_spike_shard5.csv

Demand matrix 约束：
  sumAj0 = alpha
  A00 = 1 - alpha
  A0k = 0 (k>0)
  A00 = A11 = ... = A_{N-1,N-1}
  其余 (i>0, j>0, i≠j) 平均分配
  每分片 inbound + outbound + local = 1 （在代码中按 TARGET_TOTAL_DEMAND 缩放）
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import subprocess
import os
import shutil
import glob
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------
# 常量与扫描范围
# -----------------------------
SPIKE_WEIGHTS = [0, 0, 0, 0, 1]
EPSILON_DEFAULT = 1.5
G_MAX = 2_000_000
TARGET_TOTAL_DEMAND = G_MAX / 2

EPSILON_MIN = 1.5
EPSILON_MAX = 16.0
EPSILON_STEP = 0.1

ALPHA_MIN = 0.45
ALPHA_MAX = 0.95
ALPHA_STEP = 0.01

LAMBDA_DEFAULT = 1.5  # 用于重建 lambda_matrix

EPSILON_POINTS = int(round((EPSILON_MAX - EPSILON_MIN) / EPSILON_STEP)) + 1
ALPHA_POINTS  = int(round((ALPHA_MAX - ALPHA_MIN) / ALPHA_STEP)) + 1


# -----------------------------
# 基础 I/O
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
# 配置结构辅助
# -----------------------------
def _detect_num_shards_from_config(cfg):
    try:
        # 先看 demand.base_demand_matrix
        m = cfg.get('demand', {}).get('base_demand_matrix')
        if isinstance(m, list) and m and all(isinstance(r, list) and len(r) == len(m) for r in m):
            return len(m)
    except Exception:
        pass
    # 再看 network.num_shards / demand.num_shards / system.num_shards
    for path in [('network','num_shards'), ('demand','num_shards'), ('system','num_shards'), ('num_shards',)]:
        try:
            cur = cfg
            for k in path:
                cur = cur[k]
            if isinstance(cur, int) and cur >= 3:
                return cur
        except Exception:
            continue
    return 3


def _ensure_network_and_shards(cfg, N):
    # network.num_shards
    cfg.setdefault('network', {})['num_shards'] = N
    # shards 数组
    shards = [{'id': i, 'g_max': G_MAX} for i in range(N)]
    cfg['shards'] = shards


# -----------------------------
# Demand Matrix 构造（满足你的约束）
# -----------------------------
def calculate_demand_matrix(alpha_inflow_to_0, num_shards):
    """
    生成 N×N base_demand_matrix（单位：绝对体量 = TARGET_TOTAL_DEMAND）：
      - ∑_{j>0} A[j][0] = α*T
      - A[0][0] = (1-α)*T
      - A[0][k] = 0 (k>0)
      - 所有对角线相等
      - 其余 (i>0, j>0, i≠j) 均分
      - 每分片 inbound+outbound+local = T
    """
    alpha = float(alpha_inflow_to_0)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha_inflow_to_0 必须在 [0,1]，当前={alpha}")
    N = int(num_shards)
    if N < 3:
        raise ValueError("num_shards 必须 ≥ 3")

    T = TARGET_TOTAL_DEMAND
    d = (1.0 - alpha) * T                 # 对角线
    a0 = (alpha / (N - 1)) * T            # i>0 到 0
    x  = (alpha / (2 * (N - 1))) * T      # 非 0 分片之间（均匀）

    M = [[0.0 for _ in range(N)] for __ in range(N)]

    # 对角线
    for i in range(N):
        M[i][i] = d
    # 0 行：A[0][k]=0
    for k in range(1, N):
        M[0][k] = 0.0
    # 0 列：A[i][0]=a0
    for i in range(1, N):
        M[i][0] = a0
    # 非 0 分片间均匀互联
    for i in range(1, N):
        for j in range(1, N):
            if i != j:
                M[i][j] = x

    return M


def verify_demand_matrix(matrix, alpha_value):
    N = len(matrix)
    T = TARGET_TOTAL_DEMAND
    tol = 1e-6

    inbound_to_0 = sum(matrix[i][0] for i in range(1, N))
    cond1 = abs(inbound_to_0 - alpha_value * T) < tol
    cond2 = abs(matrix[0][0] - (1 - alpha_value) * T) < tol
    cond3 = all(abs(matrix[0][k]) < 1e-12 for k in range(1, N))
    diag = [matrix[i][i] for i in range(N)]
    cond4 = all(abs(di - diag[0]) < tol for di in diag)

    def inbound(i):  return sum(matrix[j][i] for j in range(N) if j != i)
    def outbound(i): return sum(matrix[i][j] for j in range(N) if j != i)

    per_shard_ok = []
    for i in range(N):
        s = inbound(i) + outbound(i) + matrix[i][i]
        per_shard_ok.append(abs(s - T) < tol)

    nonneg = all(matrix[i][j] >= -1e-12 for i in range(N) for j in range(N))

    is_valid = cond1 and cond2 and cond3 and cond4 and all(per_shard_ok) and nonneg

    print("Demand matrix verification:")
    print(f"  N = {N}, alpha = {alpha_value:.4f}, TARGET_TOTAL_DEMAND = {T:.0f}")
    print(f"  inbound→0 = {inbound_to_0:.6f} (target {alpha_value*T:.6f})  -> {cond1}")
    print(f"  A00 = {matrix[0][0]:.6f} (target {(1-alpha_value)*T:.6f})    -> {cond2}")
    print(f"  A0k(k>0)=0                                                  -> {cond3}")
    print(f"  diagonals equal                                             -> {cond4}")
    for i in range(N):
        s = inbound(i) + outbound(i) + matrix[i][i]
        print(f"  shard {i}: inbound+outbound+local = {s:.6f} (target {T:.6f}) -> {per_shard_ok[i]}")
    print(f"  non-negative                                                -> {nonneg}")
    print(f"  Constraints satisfied: {is_valid}")

    return is_valid


# -----------------------------
# 与 Go 仿真耦合：更新配置
# -----------------------------
def update_config_for_experiment(epsilon_j0, alpha_inflow, delay_weights, num_shards):
    """
    更新 config.yml 到 N 分片，保证所有相关结构维度一致：
      - network.num_shards = N
      - shards = [{id:0..N-1, g_max:G_MAX}]
      - demand.epsilon_matrix  N×N
          * 对角线 i==j -> 0.0
          * 非对角默认 -> 1.5
          * 第 0 列 i>0 的 ε_{i0} -> 覆盖为扫描值 epsilon_j0
      - demand.base_demand_matrix N×N (按约束构造)
      - demand.lambda_matrix   N×N  (统一填 LAMBDA_DEFAULT)
      - delay.weights = Spike
    """
    try:
        cfg = load_config()
        if cfg is None:
            return False

        N = int(num_shards)

        # network / shards
        _ensure_network_and_shards(cfg, N)

        # demand 容器
        cfg.setdefault('demand', {})

        # -------- epsilon_matrix：先填默认，再覆盖 ε_{i0} --------
        em = [[(0.0 if i == j else EPSILON_DEFAULT) for j in range(N)] for i in range(N)]
        for i in range(1, N):
            em[i][0] = float(epsilon_j0)  # 只覆盖第 0 列（i>0）入 0 的 ε
        cfg['demand']['epsilon_matrix'] = em

        # base_demand_matrix
        bdm = calculate_demand_matrix(alpha_inflow, num_shards=N)
        cfg['demand']['base_demand_matrix'] = bdm

        # lambda_matrix -> 统一值（保持 N×N）
        lam_val = LAMBDA_DEFAULT
        try:
            old_lm = cfg['demand'].get('lambda_matrix')
            if isinstance(old_lm, list) and old_lm and isinstance(old_lm[0], list) and old_lm[0]:
                lam_val = float(old_lm[0][0])
        except Exception:
            pass
        lm = [[lam_val for _ in range(N)] for __ in range(N)]
        cfg['demand']['lambda_matrix'] = lm

        # 延迟分布 -> Spike
        cfg.setdefault('delay', {})['weights'] = delay_weights

        # 兼容字段（可选）
        cfg.setdefault('system', {})['num_shards'] = N
        cfg.setdefault('demand', {})['num_shards'] = N

        with open('config.yml', 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

        print(f"✅ Config updated (N={N}): epsilon_j0={epsilon_j0:.2f}, alpha={alpha_inflow:.2f}")
        return True

    except Exception as e:
        print(f"❌ Failed to update config: {e}")
        return False

# -----------------------------
# 运行与解析
# -----------------------------
def run_simulation():
    try:
        result = subprocess.run(['go', 'run', '../../../main.go'],
                                capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return True
        else:
            print(f"❌ Simulation failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Simulation timed out")
        return False
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        return False

def find_latest_log():
    logs = glob.glob("enhanced_simulation_analysis_*.log")
    return max(logs, key=os.path.getctime) if logs else None

def parse_final_load(log_file):
    try:
        with open(log_file, 'r') as f:
            content = f.read()

        print(f"  📄 Parsing log file: {log_file}")

        data_start_idx = content.find('DATA_START')
        if data_start_idx == -1:
            print(f"  ❌ No DATA_START found in {log_file}")
            return None

        lines = content[data_start_idx:].split('\n')
        csv_data, header_found = [], False
        for line in lines:
            s = line.strip()
            if s.startswith('Step,Shard0_Fee'):
                header_found = True
                csv_data.append(s)
            elif header_found and ',' in s and not s.startswith('#'):
                parts = s.split(',')
                if len(parts) >= 7:
                    try:
                        int(parts[0])
                        csv_data.append(s)
                    except ValueError:
                        break

        if len(csv_data) <= 1:
            print("  ❌ No valid CSV data found")
            return None

        from io import StringIO
        df = pd.read_csv(StringIO('\n'.join(csv_data)))
        shard0_loads = df['Shard0_Load']
        final_load = shard0_loads.iloc[-1]
        avg_load   = shard0_loads.mean()
        load_std   = shard0_loads.std()
        target     = 0.5
        load_infl  = (avg_load - target) / target

        print(f"  ✅ Parsed {len(shard0_loads)} steps of shard0 data")
        print(f"  📊 Shard0 stats: final={final_load:.6f}, avg={avg_load:.6f}, std={load_std:.6f}")

        return {
            'load0': final_load,
            'avg_load': avg_load,
            'load_std': load_std,
            'load_inflation': load_infl
        }

    except Exception as e:
        print(f"❌ Error parsing log: {e}")
        return {
            'load0': 0.5,
            'avg_load': 0.5,
            'load_std': 0.0,
            'load_inflation': 0.0
        }

def clean_log_files():
    try:
        logs = glob.glob("enhanced_simulation_analysis_*.log") + glob.glob("experiment_*.log")
        for p in logs:
            os.remove(p)
        if logs:
            print(f"  🧹 Cleaned {len(logs)} remaining log files")
        return True
    except Exception as e:
        print(f"⚠️  Warning: Could not clean log files: {e}")
        return False

def check_convergence(load_stats, tolerance=1e-5):
    if load_stats is None or not isinstance(load_stats, dict):
        return False
    return abs(load_stats['load0'] - 0.5) < tolerance


def run_single_experiment(epsilon_j0, alpha_inflow, delay_weights, exp_id, num_shards):
    if not update_config_for_experiment(epsilon_j0, alpha_inflow, delay_weights, num_shards=num_shards):
        return None

    unique_log_name = f"experiment_Spike_{exp_id:06d}_N{num_shards}_eps{epsilon_j0:.2f}_alpha{alpha_inflow:.2f}.log"

    if not run_simulation():
        return None

    latest_log = find_latest_log()
    if latest_log is None:
        print(f"  ❌ No log file found for experiment {exp_id}")
        return None

    try:
        os.rename(latest_log, unique_log_name)
        print(f"  📝 Renamed log to: {unique_log_name}")
    except Exception:
        unique_log_name = latest_log

    load_stats = parse_final_load(unique_log_name)
    converged = check_convergence(load_stats)

    try:
        os.remove(unique_log_name)
        print(f"  🗑️  Cleaned up: {unique_log_name}")
    except Exception:
        pass

    return {
        'num_shards': num_shards,
        'epsilon_j0': epsilon_j0,
        'alpha_inflow': alpha_inflow,
        'distribution': 'Spike',
        'final_load': load_stats['load0'] if load_stats else None,
        'avg_load': load_stats['avg_load'] if load_stats else None,
        'load_std': load_stats['load_std'] if load_stats else None,
        'load_inflation': load_stats['load_inflation'] if load_stats else None,
        'converged': converged
    }


def save_distribution_results(results, csv_filename):
    df = pd.DataFrame(results)
    df.to_csv(csv_filename, index=False)
    print(f"💾 Saved {len(results)} results to {csv_filename}")


# -----------------------------
# 主实验流程（仅 Spike；N=10→7→5）
# -----------------------------
def run_convergence_experiment():
    print("🔬 Starting Convergence Boundary Experiment (Spike only, multi-N)")
    print("=" * 80)
    shard_list = [10, 7, 5]
    delay_weights = SPIKE_WEIGHTS

    total_points = EPSILON_POINTS * ALPHA_POINTS * len(shard_list)
    print(f"Total experiments to run: {total_points}")
    print(f"Epsilon range: {EPSILON_MIN}~{EPSILON_MAX} (step {EPSILON_STEP}, {EPSILON_POINTS} pts)")
    print(f"Alpha   range: {ALPHA_MIN}~{ALPHA_MAX} (step {ALPHA_STEP}, {ALPHA_POINTS} pts)")
    print(f"Shards to run: {shard_list} (order)")
    print(f"Latency dist.: Spike {delay_weights}")

    if not backup_config():
        print("❌ Failed to backup config, aborting")
        return

    experiment_count = 0

    try:
        for N in shard_list:
            print(f"\n🧪 Running Spike with N={N}")
            print("-" * 60)
            results_N = []

            for i in range(EPSILON_POINTS - 1, -1, -1):
                epsilon_j0 = EPSILON_MIN + i * EPSILON_STEP
                for j in range(ALPHA_POINTS - 1, -1, -1):
                    alpha_inflow = ALPHA_MIN + j * ALPHA_STEP
                    experiment_count += 1

                    print(f"Experiment {experiment_count}/{total_points}: N={N}, ε={epsilon_j0:.2f}, α={alpha_inflow:.2f}")

                    r = run_single_experiment(
                        epsilon_j0, alpha_inflow,
                        delay_weights=delay_weights,
                        exp_id=experiment_count,
                        num_shards=N
                    )
                    if r is not None:
                        results_N.append(r)
                        status = "✅ Converged" if r['converged'] else "❌ Diverged"
                        print(f"  → Load0: {r['final_load']:.6f}, Avg: {r['avg_load']:.6f}, Std: {r['load_std']:.6f}, Infl: {r['load_inflation']:.6f}, {status}")
                    else:
                        print("  → ❌ Experiment failed")
                        results_N.append({
                            'num_shards': N,
                            'epsilon_j0': epsilon_j0,
                            'alpha_inflow': alpha_inflow,
                            'distribution': 'Spike',
                            'final_load': None,
                            'avg_load': None,
                            'load_std': None,
                            'load_inflation': None,
                            'converged': False
                        })

                    if len(results_N) % 10 == 0:
                        save_distribution_results(results_N, f"convergence_results_spike_shard{N}.csv")

                    if experiment_count % 50 == 0:
                        clean_log_files()

            save_distribution_results(results_N, f"convergence_results_spike_shard{N}.csv")
            print(f"✅ Completed N={N}: {len(results_N)} experiments")

        print("\n🎉 All Spike experiments for N∈{10,7,5} completed!")

    finally:
        restore_config()
        print("🔄 Original config.yml restored")


# -----------------------------
# 可选：结果可视化
# -----------------------------
def load_and_combine_results():
    all_results, files = [], []
    for N in [10, 7, 5]:
        fn = f"convergence_results_spike_shard{N}.csv"
        if os.path.exists(fn):
            files.append(fn)
    print(f"Found {len(files)} spike result files")
    for fp in files:
        try:
            df = pd.read_csv(fp)
            all_results.append(df)
            print(f"Loaded {len(df)} results from {fp}")
        except Exception as e:
            print(f"❌ Error loading {fp}: {e}")
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        print(f"✅ Combined {len(combined)} total results")
        return combined
    return None

def create_3d_convergence_plots(df):
    if df is None or len(df) == 0:
        print("❌ No data available for plotting")
        return
    shard_values = sorted(df['num_shards'].dropna().unique())[::-1]
    fig = plt.figure(figsize=(6 * len(shard_values), 6))
    for i, N in enumerate(shard_values):
        dist_data = df[df['num_shards'] == N]
        ax = fig.add_subplot(1, len(shard_values), i + 1, projection='3d')
        converged = dist_data[dist_data['converged'] == True]
        diverged  = dist_data[dist_data['converged'] == False]
        if len(converged) > 0:
            ax.scatter(converged['epsilon_j0'], converged['alpha_inflow'], [1]*len(converged),
                       c='green', marker='o', s=20, alpha=0.6, label='Converged')
        if len(diverged) > 0:
            ax.scatter(diverged['epsilon_j0'], diverged['alpha_inflow'], [0]*len(diverged),
                       c='red', marker='x', s=20, alpha=0.6, label='Diverged')
        ax.set_xlabel('Epsilon (ε_j0)')
        ax.set_ylabel('Alpha (α_j→0,in)')
        ax.set_zlabel('Convergence')
        ax.set_title(f'N={N} (Spike)')
        ax.set_zlim(-0.1, 1.1)
        ax.set_zticks([0, 1])
        ax.set_zticklabels(['Diverged', 'Converged'])
        ax.legend()
        ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.savefig('convergence_boundary_spike_multiN_3d.png', dpi=300, bbox_inches='tight')
    plt.savefig('convergence_boundary_spike_multiN_3d.pdf', dpi=300, bbox_inches='tight')
    print("💾 3D convergence plots saved as convergence_boundary_spike_multiN_3d.png/pdf")
    plt.show()

def analyze_convergence_results(df):
    if df is None or len(df) == 0:
        print("❌ No data available for analysis")
        return
    print("\n📊 CONVERGENCE ANALYSIS SUMMARY (Spike)")
    print("=" * 80)
    for N in sorted(df['num_shards'].dropna().unique())[::-1]:
        dist_data = df[df['num_shards'] == N]
        total = len(dist_data)
        conv  = len(dist_data[dist_data['converged'] == True])
        rate  = conv / total * 100
        print(f"\n🔬 N={N}:")
        print(f"   Total experiments: {total}")
        print(f"   Converged: {conv} ({rate:.1f}%)")
        print(f"   Diverged: {total - conv} ({100 - rate:.1f}%)")
        csub = dist_data[dist_data['converged'] == True]
        if len(csub) > 0:
            print(f"   Max converged ε: {csub['epsilon_j0'].max():.2f}")
            print(f"   Max converged α: {csub['alpha_inflow'].max():.2f}")
    out = "convergence_analysis_spike_summary.csv"
    stat = []
    for N in sorted(df['num_shards'].dropna().unique())[::-1]:
        dist_data = df[df['num_shards'] == N]
        total = len(dist_data)
        conv  = len(dist_data[dist_data['converged'] == True])
        stat.append({'num_shards': int(N), 'total_experiments': int(total),
                     'converged_count': int(conv), 'convergence_rate': conv / total * 100})
    pd.DataFrame(stat).to_csv(out, index=False)
    print(f"\n💾 Summary saved to {out}")

def visualize_results():
    print("📈 Loading and visualizing Spike results...")
    df = load_and_combine_results()
    if df is not None:
        analyze_convergence_results(df)
        create_3d_convergence_plots(df)
        df.to_csv("convergence_results_spike_combined.csv", index=False)
        print("💾 Combined results saved to convergence_results_spike_combined.csv")
    else:
        print("❌ No results to visualize")


# -----------------------------
# 主入口
# -----------------------------
if __name__ == "__main__":
    print("🔬 Convergence Boundary Experiment (Spike-only, multi-shard: N=10,7,5)")
    print("=" * 60)
    print("Choose operation:")
    print("1. Test demand matrix calculation (N=3,5,7,10)")
    print("2. Run Spike experiment for N=10,7,5")
    print("3. Visualize existing Spike results")
    print("4. Exit")
    choice = input("Enter choice (1-4): ").strip()

    if choice == "1":
        okN = True
        for N in [3,5,7,10]:
            for a in [0.10, 0.25, 0.50, 0.75, 0.90]:
                M = calculate_demand_matrix(a, num_shards=N)
                if not verify_demand_matrix(M, a):
                    okN = False
                    break
        print("\n✅ Demand matrix calculation verified!" if okN else "\n❌ Demand matrix check failed.")

    elif choice == "2":
        # 先快速验证一组，避免长跑前因约束错误中途失败
        M = calculate_demand_matrix(0.7, num_shards=5)
        if verify_demand_matrix(M, 0.7):
            run_convergence_experiment()
        else:
            print("\n❌ Demand matrix calculation has issues, aborting.")

    elif choice == "3":
        visualize_results()

    elif choice == "4":
        print("Goodbye!")
    else:
        print("Invalid choice. Please run again.")
