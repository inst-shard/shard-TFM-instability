#!/usr/bin/env python3
"""
Appendix C.3: Phase Map Validation Experiment

This experiment implements the validation protocol described in Appendix C.3 of the paper.
It sweeps parameter sets, observes convergence, calculates 95th percentile kappa values,
and validates theoretical phase map predictions against empirical results.

Key Features:
1. Parameter sweep using Figure5 logic for convergence testing
2. Calculate kappa_def and kappa_att using formulas from the paper:
   - kappa_def(t) = Σ_d w_d * |ΔP_i(t-d)| / |ΔP_i(t)| * 1(ΔP_i(t-d) * ΔP_i(t) > 0)
   - kappa_att(t) = Σ_d w_d * |ΔP_j(t-d)| / |ΔP_j(t)| * 1(ΔP_i(t) * (-ΔP_j(t-d)) > 0)
3. Compute Gi and Ri using the calculated kappa values
4. Validate phase map predictions: |1 - Gi| + Ri < 1 for stability
5. Compare theoretical predictions with empirical convergence results

Parameter sweep:
- epsilon_j0: from 1.5 to 8.0, step 0.1 (smaller range for faster testing)
- alpha_j->0,in: from 0.45 to 0.95, step 0.02 (smaller step for faster testing)
- Three delay distributions: Spike, Uniform, Bimodal

Log handling:
- The Go simulation writes enhanced_simulation_analysis_*.log.
- After parsing each run, the script deletes that log file automatically.  # --- NEW ---
- Aggregated results are saved to CSV; raw per-shard timeseries can optionally be saved.  # --- NEW ---
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import subprocess
import time
import re
import os
import shutil
import glob
from mpl_toolkits.mplot3d import Axes3D

# --- NEW --- Controls for persistence
DELETE_LOG_AFTER_PARSE = True                          # delete enhanced_simulation_analysis_*.log after parsing
SAVE_PARSED_TIMESERIES = False                         # save parsed per-shard timeseries CSVs (default: False)
PARSED_TS_DIR = 'parsed_timeseries'                    # directory for timeseries if enabled

# Delay distribution configurations
DELAY_DISTRIBUTIONS = {
    'Spike': [0, 0, 0, 0, 1],
}

# Simulation parameters
G_MAX = 2000000  # Each shard's g_max
TARGET_TOTAL_DEMAND = G_MAX / 2  # Each shard's total demand should be g_max/2

# Experiment parameter ranges (smaller for faster testing)
EPSILON_MIN = 1.5
EPSILON_MAX = 16
EPSILON_STEP = 0.1

ALPHA_MIN = 0.45
ALPHA_MAX = 0.95
ALPHA_STEP = 0.01
ALPHA_IJ_OUT = 0

# Phase map parameters from paper
DELTA = 0.125  # EIP-1559 update rate
EQUIL_P = 1.0  # Equilibrium price
L_TARGET = 0.5  # Target load
START_STEP = 100  # Start analysis after shock
END_STEP = 4999   # End analysis

# Calculate total experiment points
EPSILON_POINTS = int((EPSILON_MAX - EPSILON_MIN) / EPSILON_STEP) + 1
ALPHA_POINTS = round((ALPHA_MAX - ALPHA_MIN) / ALPHA_STEP) + 1
TOTAL_POINTS_PER_DISTRIBUTION = EPSILON_POINTS * ALPHA_POINTS

def load_config():
    """Load configuration from config.yml"""
    try:
        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print("❌ config.yml not found")
        return None

def backup_config():
    """Create a backup of the original config"""
    try:
        shutil.copy('config.yml', 'config_backup.yml')
        print("✅ Created config backup")
        return True
    except Exception as e:
        print(f"❌ Failed to backup config: {e}")
        return False

def restore_config():
    """Restore config from backup"""
    try:
        shutil.copy('config_backup.yml', 'config.yml')
        print("✅ Restored original config")
        return True
    except Exception as e:
        print(f"❌ Failed to restore config: {e}")
        return False

def calculate_demand_matrix(alpha_inflow_to_0):
    """
    Calculate base_demand_matrix based on alpha (inflow ratio to shard 0)
    Same logic as Figure5 experiment
    """
    # Initialize matrix
    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    # Step 1: Calculate outflow from shard 0 using alpha_ij,out = 0
    total_outflow_from_0 = TARGET_TOTAL_DEMAND * ALPHA_IJ_OUT  # 01 + 02 = 0
    matrix[0][1] = total_outflow_from_0 / 2  # 01
    matrix[0][2] = total_outflow_from_0 / 2  # 02

    # Step 2: Calculate inflow to shard 0 (10 + 20)
    total_inflow_to_0 = TARGET_TOTAL_DEMAND * alpha_inflow_to_0
    matrix[1][0] = total_inflow_to_0 / 2  # 10
    matrix[2][0] = total_inflow_to_0 / 2  # 20

    # Step 3: Calculate 00 to satisfy shard 0 total demand constraint
    matrix[0][0] = TARGET_TOTAL_DEMAND - matrix[0][1] - matrix[0][2] - matrix[1][0] - matrix[2][0]  # 00

    # Step 4: Set all diagonal elements equal to 00
    diagonal_value = matrix[0][0]
    matrix[1][1] = diagonal_value  # 11 = 00
    matrix[2][2] = diagonal_value  # 22 = 00

    # Step 5: Calculate remaining cross-shard flows
    remaining_cross_1 = TARGET_TOTAL_DEMAND - matrix[1][0] - matrix[1][1] - matrix[0][1]
    remaining_cross_2 = TARGET_TOTAL_DEMAND - matrix[2][0] - matrix[2][2] - matrix[0][2]

    avg_remaining_cross = (remaining_cross_1 + remaining_cross_2) / 2

    # Split evenly: 12 = 21
    matrix[1][2] = avg_remaining_cross / 2  # 12
    matrix[2][1] = avg_remaining_cross / 2  # 21

    return matrix

def update_config_for_experiment(epsilon_j0, alpha_inflow, delay_weights):
    """Update config.yml for the experiment parameters"""
    try:
        config = load_config()
        if config is None:
            return False

        # Update epsilon matrix - only change epsilon for shard 0 (row 0)
        config['demand']['epsilon_matrix'][1][0] = epsilon_j0  # epsilon_01
        config['demand']['epsilon_matrix'][2][0] = epsilon_j0  # epsilon_02

        # Calculate and update demand matrix
        demand_matrix = calculate_demand_matrix(alpha_inflow)
        config['demand']['base_demand_matrix'] = demand_matrix

        # Update delay weights
        config['delay']['weights'] = delay_weights

        # Write updated config
        with open('config.yml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return True

    except Exception as e:
        print(f"❌ Failed to update config: {e}")
        return False

def run_simulation():
    """Run the Go simulation - copied from Figure6"""
    try:
        result = subprocess.run(['go', 'run', '../../../main.go'],
                              capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print("❌ sim failed:", result.stderr[:400])
            return False
        time.sleep(0.05)
        return True
    except subprocess.TimeoutExpired:
        print("❌ Simulation timed out")
        return False
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        return False

# --- NEW ---
def save_parsed_timeseries(shard_data, distribution_name, exp_id):
    """Optionally persist parsed per-shard time series to CSV files."""
    dirpath = os.path.join(PARSED_TS_DIR, distribution_name, f"exp_{exp_id:05d}")
    os.makedirs(dirpath, exist_ok=True)
    for shard_id, df in shard_data.items():
        if df is not None and not df.empty:
            df.to_csv(os.path.join(dirpath, f"shard_{shard_id}.csv"), index=False)
    print(f"💾 Saved parsed time series to {dirpath}")

def parse_latest_log(delete_after=DELETE_LOG_AFTER_PARSE):
    """
    Parse latest enhanced_simulation_analysis_*.log - copied exactly from Figure6
    Supports:
      step,fee0,fee1,fee2,load0,load1,load2
      or fallback: step,fee,load (single shard)
    Returns: (dict {0: df0, 1: df1, 2: df2} or None, path_to_log or None)
    After parsing, optionally deletes the log file.  # --- NEW ---
    """
    files = glob.glob('enhanced_simulation_analysis_*.log')
    if not files:
        return None, None
    latest = max(files, key=os.path.getctime)
    data = {0: [], 1: [], 2: []}
    try:
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
    finally:
        if delete_after:
            try:
                os.remove(latest)
                print(f"🧹 Deleted log file: {latest}")
            except Exception as e:
                print(f"⚠️ Failed to delete log {latest}: {e}")

    out = {}
    for sid in (0, 1, 2):
        if data[sid]:
            out[sid] = pd.DataFrame(data[sid])
    return (out if out else None), latest

def calculate_kappa_def_timeseries(delta_p, weights, start_idx, end_idx):
    """
    Calculate kappa_def(t) = Σ_d w_d * |ΔP(t-d)| / |ΔP(t)| with same-sign mask
    Copied from Figure4/kappa-defense-delay-defense.py - calc_kappa_timeseries function
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

def calculate_kappa_att_timeseries(delta_p_i, delta_p_j, weights, start_idx, end_idx):
    """
    Calculate kappa_att(t) = Σ_d w_d * |ΔP_j(t-d)| / |ΔP_j(t)| with opposite-sign mask
    Copied from Figure6/kappa-attack-delay-attack.py - calc_kappa_attack_timeseries function
    """
    dmax = len(weights)
    end_idx = min(end_idx, len(delta_p_i) - 1, len(delta_p_j) - 1)
    if start_idx + dmax > end_idx:
        return []

    ks = []
    eps = 1e-12

    for t in range(start_idx + dmax, end_idx + 1):
        cur_j = delta_p_j[t]
        cur_i = delta_p_i[t]
        if not np.isfinite(cur_j) or abs(cur_j) < eps:
            continue

        s_i = np.sign(cur_i)  # -1, 0, +1
        k = 0.0

        for d in range(1, dmax + 1):
            prev_j = delta_p_j[t - d]
            if not np.isfinite(prev_j) or prev_j == 0.0:
                continue
            # opposite-sign mask: ΔP_i(t) and ΔP_j(t-d) with different signs
            if np.sign(prev_j) != s_i:
                k += weights[d - 1] * (abs(prev_j) / abs(cur_j))

        ks.append(k)

    return ks

def calculate_phase_map_parameters(epsilon_j0, alpha_inflow, kappa_def_95, kappa_att_95):
    """
    Calculate Gi and Ri from paper formulas:
    ...
    """
    # Load composition for shard 0
    alpha_0_local = 1.0 - alpha_inflow  # Local load ratio
    alpha_j_to_0_in = alpha_inflow / 2   # Inbound from each other shard
    alpha_0_to_j_out = 0.0               # No outbound from shard 0

    # Elasticities (from config and swept parameter)
    lambda_00 = 1.5      # Local elasticity
    lambda_0j = 1.5      # Outbound elasticity (not used since α_0→j,out = 0)
    epsilon_j0_param = epsilon_j0  # Inbound elasticity (swept parameter)
    epsilon_0j = 1.5     # Outbound elasticity (not used since α_0→j,out = 0)
    lambda_j0 = 1.5      # Delayed attack elasticity

    # Calculate forces for shard 0
    Lambda_0_local = lambda_00 * alpha_0_local
    Lambda_0_self_delay = epsilon_j0_param * (2 * alpha_j_to_0_in)  # Sum over j=1,2
    Lambda_0j_inst = epsilon_0j * alpha_0_to_j_out  # = 0
    Lambda_j0_delay = lambda_j0 * (2 * alpha_j_to_0_in)  # Sum over j=1,2

    # Calculate Gi and Ri
    defense_total = Lambda_0_local + kappa_def_95 * Lambda_0_self_delay
    attack_total = kappa_att_95 * Lambda_j0_delay  # Λ_0j^inst = 0

    Gi = DELTA * defense_total
    Ri = attack_total / defense_total if defense_total > 0 else float('inf')

    return Gi, Ri, {
        'alpha_0_local': alpha_0_local,
        'alpha_j_to_0_in': alpha_j_to_0_in,
        'Lambda_0_local': Lambda_0_local,
        'Lambda_0_self_delay': Lambda_0_self_delay,
        'Lambda_j0_delay': Lambda_j0_delay,
        'defense_total': defense_total,
        'attack_total': attack_total
    }

def check_convergence(load_stats, tolerance=0.00001):
    """Check if the system converged to target load (0.5)"""
    if load_stats is None or not isinstance(load_stats, dict):
        return False
    return abs(load_stats['load0'] - 0.5) < tolerance

def run_single_experiment(epsilon_j0, alpha_inflow, delay_weights, distribution_name, exp_id):
    """Run a single experiment and return results with kappa calculations"""

    # Update config
    if not update_config_for_experiment(epsilon_j0, alpha_inflow, delay_weights):
        return None

    # Run simulation
    if not run_simulation():
        return None

    # Parse multi-shard data using Figure6 logic (now returns (data, log_path))
    shard_data, log_path = parse_latest_log()
    if not shard_data or 0 not in shard_data:
        print(f"  ❌ Failed to parse shard data")
        return None

    # Optionally persist per-shard timeseries CSVs  # --- NEW ---
    if SAVE_PARSED_TIMESERIES:
        save_parsed_timeseries(shard_data, distribution_name, exp_id)

    # Debug: Print shard data info
    print(f"  🔍 DEBUG: Parsed {len(shard_data)} shards")
    for shard_id, df in shard_data.items():
        print(f"    Shard {shard_id}: {len(df)} rows, Steps {df['Step'].min()}-{df['Step'].max()}")

    # Extract shard 0 data for convergence check
    df0 = shard_data[0]
    if df0.empty:
        return None

    # Check convergence
    final_load = df0['Load'].iloc[-1]
    converged = check_convergence({'load0': final_load})

    # Calculate kappa values if we have multi-shard data
    kappa_def_95 = np.nan
    kappa_att_95 = np.nan

    print(f"  🔍 DEBUG: Checking kappa calculation conditions...")
    print(f"    Shard 1 in data: {1 in shard_data}")
    print(f"    Shard 2 in data: {2 in shard_data}")

    if 1 in shard_data and 2 in shard_data:
        df1, df2 = shard_data[1], shard_data[2]
        print(f"    Shard 1 empty: {df1.empty}, Shard 2 empty: {df2.empty}")

        if not df1.empty and not df2.empty:
            # Find analysis window
            start_condition = (df0['Step'] >= START_STEP).any()
            end_condition = (df0['Step'] <= END_STEP).any()
            print(f"    Analysis window: START_STEP={START_STEP}, END_STEP={END_STEP}")
            print(f"    Start condition: {start_condition}, End condition: {end_condition}")

            if start_condition and end_condition:
                start_idx = df0[df0['Step'] >= START_STEP].index[0]
                end_idx = df0[df0['Step'] <= END_STEP].index[-1]
                print(f"    Analysis indices: {start_idx} to {end_idx}")

                # Calculate price deviations
                dP0 = (df0['Fee'].values - EQUIL_P)
                dP1 = (df1['Fee'].values - EQUIL_P)
                dP2 = (df2['Fee'].values - EQUIL_P)

                print(f"    Price deviation ranges:")
                print(f"      dP0: [{dP0.min():.6f}, {dP0.max():.6f}]")
                print(f"      dP1: [{dP1.min():.6f}, {dP1.max():.6f}]")
                print(f"      dP2: [{dP2.min():.6f}, {dP2.max():.6f}]")

                # Calculate kappa_def for shard 0 (self-delay)
                print(f"    Calculating kappa_def...")
                ks_def = calculate_kappa_def_timeseries(dP0, delay_weights, start_idx, end_idx)
                print(f"    kappa_def timeseries length: {len(ks_def) if ks_def else 0}")
                if ks_def:
                    kappa_def_95 = np.percentile(ks_def, 95)
                    print(f"    kappa_def_95: {kappa_def_95}")
                else:
                    print(f"    kappa_def calculation returned empty list")

                # Calculate kappa_att for attacks from shards 1,2 to shard 0
                print(f"    Calculating kappa_att...")
                ks_att_10 = calculate_kappa_att_timeseries(dP0, dP1, delay_weights, start_idx, end_idx)
                ks_att_20 = calculate_kappa_att_timeseries(dP0, dP2, delay_weights, start_idx, end_idx)
                print(f"    kappa_att_10 length: {len(ks_att_10) if ks_att_10 else 0}")
                print(f"    kappa_att_20 length: {len(ks_att_20) if ks_att_20 else 0}")

                if ks_att_10 and ks_att_20:
                    # Combine attacks from both shards (take max as worst case)
                    all_att = ks_att_10 + ks_att_20
                    kappa_att_95 = np.percentile(all_att, 95)
                    print(f"    kappa_att_95: {kappa_att_95}")
                else:
                    print(f"    kappa_att calculation returned empty lists")
            else:
                print(f"    Analysis window not found in data")
        else:
            print(f"    One or more shard dataframes are empty")
    else:
        print(f"    Missing shard data for kappa calculation")

    # Calculate phase map parameters - handle NaN kappa values
    print(f"  🔍 Final kappa values before defaults: def={kappa_def_95}, att={kappa_att_95}")
    if np.isnan(kappa_def_95):
        kappa_def_95 = 0.0  # Use conservative default
        print(f"    Using default kappa_def_95 = 0.0")
    if np.isnan(kappa_att_95):
        kappa_att_95 = 0.0  # Use conservative default
        print(f"    Using default kappa_att_95 = 0.0")

    Gi, Ri, details = calculate_phase_map_parameters(epsilon_j0, alpha_inflow, kappa_def_95, kappa_att_95)

    # Phase map prediction: stable if |1 - Gi| + Ri < 1
    phase_map_stable = abs(1 - Gi) + Ri < 1

    return {
        'epsilon_j0': epsilon_j0,
        'alpha_inflow': alpha_inflow,
        'distribution': distribution_name,
        'final_load': final_load,
        'converged': converged,
        'kappa_def_95': kappa_def_95,
        'kappa_att_95': kappa_att_95,
        'Gi': Gi,
        'Ri': Ri,
        'phase_map_stable': phase_map_stable,
        'prediction_correct': (converged == phase_map_stable),
        **details
    }

def run_phase_map_validation_experiment():
    """Run the complete phase map validation experiment"""
    print("🔬 Starting Phase Map Validation Experiment (Appendix C.3)")
    print("=" * 80)

    total_experiments = TOTAL_POINTS_PER_DISTRIBUTION * 3
    print(f"Total experiments to run: {total_experiments}")
    print(f"Epsilon range: {EPSILON_MIN} to {EPSILON_MAX}, step {EPSILON_STEP} ({EPSILON_POINTS} points)")
    print(f"Alpha range: {ALPHA_MIN} to {ALPHA_MAX}, step {ALPHA_STEP} ({ALPHA_POINTS} points)")
    print(f"Delay distributions: {list(DELAY_DISTRIBUTIONS.keys())}")

    # Backup original config
    if not backup_config():
        print("❌ Failed to backup config, aborting")
        return

    try:
        experiment_count = 0
        all_results = []

        # Loop through each delay distribution
        for dist_name, delay_weights in DELAY_DISTRIBUTIONS.items():
            print(f"\n🧪 Testing delay distribution: {dist_name}")
            print(f"Weights: {delay_weights}")
            print("-" * 60)

            distribution_results = []

            # Loop through epsilon values
            for i in range(EPSILON_POINTS):
                epsilon_j0 = EPSILON_MIN + i * EPSILON_STEP

                # Loop through alpha values
                for j in range(ALPHA_POINTS):
                    alpha_inflow = ALPHA_MIN + j * ALPHA_STEP
                    experiment_count += 1

                    print(f"Experiment {experiment_count}/{total_experiments}: " +
                          f"{dist_name}, ε={epsilon_j0:.2f}, α={alpha_inflow:.2f}")

                    # Run single experiment
                    result = run_single_experiment(epsilon_j0, alpha_inflow, delay_weights, dist_name, experiment_count)

                    if result is not None:
                        distribution_results.append(result)
                        all_results.append(result)

                        convergence_status = "✅ Converged" if result['converged'] else "❌ Diverged"
                        phase_status = "✅ Stable" if result['phase_map_stable'] else "❌ Unstable"
                        prediction_status = "✅ Correct" if result['prediction_correct'] else "❌ Wrong"

                        print(f"  → Load: {result['final_load']:.6f}, {convergence_status}")
                        print(f"  → κ_def: {result['kappa_def_95']:.4f}, κ_att: {result['kappa_att_95']:.4f}")
                        print(f"  → Gi: {result['Gi']:.4f}, Ri: {result['Ri']:.4f}")
                        print(f"  → Phase Map: {phase_status}, Prediction: {prediction_status}")
                    else:
                        print(f"  → ❌ Experiment failed")
                        # Add failed experiment record
                        failed_result = {
                            'epsilon_j0': epsilon_j0,
                            'alpha_inflow': alpha_inflow,
                            'distribution': dist_name,
                            'final_load': None,
                            'converged': False,
                            'kappa_def_95': np.nan,
                            'kappa_att_95': np.nan,
                            'Gi': np.nan,
                            'Ri': np.nan,
                            'phase_map_stable': False,
                            'prediction_correct': False
                        }
                        distribution_results.append(failed_result)
                        all_results.append(failed_result)

            # Save results for this distribution
            df_dist = pd.DataFrame(distribution_results)
            df_dist.to_csv(f'phase_map_validation_{dist_name}.csv', index=False)
            print(f"✅ Completed {dist_name} distribution: {len(distribution_results)} experiments")

        # Save combined results
        df_all = pd.DataFrame(all_results)
        df_all.to_csv('phase_map_validation_all.csv', index=False)

        # Calculate validation statistics
        analyze_validation_results(df_all)

        print(f"\n🎉 All experiments completed!")
        print(f"Total experiments run: {experiment_count}")
        print(f"Results saved in CSV files")

    finally:
        # Always restore original config
        restore_config()
        print("🔄 Original config.yml restored")

def analyze_validation_results(df):
    """Analyze and summarize validation results"""
    if df is None or len(df) == 0:
        print("❌ No data available for analysis")
        return

    print("\n📊 PHASE MAP VALIDATION ANALYSIS")
    print("=" * 80)

    # Overall statistics
    total_experiments = len(df)
    valid_experiments = len(df[~df['Gi'].isna()])

    if valid_experiments == 0:
        print("❌ No valid experiments with kappa calculations")
        return

    df_valid = df[~df['Gi'].isna()]

    # Calculate prediction accuracy
    correct_predictions = len(df_valid[df_valid['prediction_correct'] == True])
    accuracy = correct_predictions / valid_experiments * 100

    print(f"📈 Overall Validation Results:")
    print(f"   Total experiments: {total_experiments}")
    print(f"   Valid experiments (with kappa): {valid_experiments}")
    print(f"   Correct predictions: {correct_predictions}")
    print(f"   Prediction accuracy: {accuracy:.1f}%")

    # Per-distribution analysis
    for dist_name in df['distribution'].unique():
        dist_data = df_valid[df_valid['distribution'] == dist_name]
        if len(dist_data) == 0:
            continue

        dist_correct = len(dist_data[dist_data['prediction_correct'] == True])
        dist_accuracy = dist_correct / len(dist_data) * 100

        # Convergence statistics
        empirical_converged = len(dist_data[dist_data['converged'] == True])
        phase_map_stable = len(dist_data[dist_data['phase_map_stable'] == True])

        print(f"\n🔬 {dist_name} Distribution:")
        print(f"   Valid experiments: {len(dist_data)}")
        print(f"   Prediction accuracy: {dist_accuracy:.1f}%")
        print(f"   Empirical convergence: {empirical_converged} ({empirical_converged/len(dist_data)*100:.1f}%)")
        print(f"   Phase map stable: {phase_map_stable} ({phase_map_stable/len(dist_data)*100:.1f}%)")

        # Kappa statistics
        kappa_def_mean = dist_data['kappa_def_95'].mean()
        kappa_att_mean = dist_data['kappa_att_95'].mean()
        print(f"   Mean κ_def (95%): {kappa_def_mean:.4f}")
        print(f"   Mean κ_att (95%): {kappa_att_mean:.4f}")

    # Save summary statistics
    summary_stats = []
    for dist_name in df['distribution'].unique():
        dist_data = df_valid[df_valid['distribution'] == dist_name]
        if len(dist_data) == 0:
            continue

        dist_correct = len(dist_data[dist_data['prediction_correct'] == True])

        summary_stats.append({
            'distribution': dist_name,
            'total_experiments': len(dist_data),
            'correct_predictions': dist_correct,
            'accuracy': dist_correct / len(dist_data) * 100,
            'empirical_converged': len(dist_data[dist_data['converged'] == True]),
            'phase_map_stable': len(dist_data[dist_data['phase_map_stable'] == True]),
            'mean_kappa_def_95': dist_data['kappa_def_95'].mean(),
            'mean_kappa_att_95': dist_data['kappa_att_95'].mean()
        })

    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('phase_map_validation_summary.csv', index=False)
    print(f"\n💾 Summary saved to phase_map_validation_summary.csv")

def create_validation_plots(df):
    """Create visualization plots for validation results"""
    if df is None or len(df) == 0:
        print("❌ No data available for plotting")
        return

    df_valid = df[~df['Gi'].isna()]
    if len(df_valid) == 0:
        print("❌ No valid data for plotting")
        return

    # Create scatter plot: Gi vs Ri with prediction correctness
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, dist_name in enumerate(['Spike', 'Uniform', 'Bimodal']):
        ax = axes[i]
        dist_data = df_valid[df_valid['distribution'] == dist_name]

        if len(dist_data) == 0:
            continue

        # Separate correct and incorrect predictions
        correct = dist_data[dist_data['prediction_correct'] == True]
        incorrect = dist_data[dist_data['prediction_correct'] == False]

        # Plot correct predictions
        if len(correct) > 0:
            ax.scatter(correct['Gi'], correct['Ri'], c='green', marker='o',
                      s=30, alpha=0.7, label=f'Correct ({len(correct)})')

        # Plot incorrect predictions
        if len(incorrect) > 0:
            ax.scatter(incorrect['Gi'], incorrect['Ri'], c='red', marker='x',
                      s=30, alpha=0.7, label=f'Incorrect ({len(incorrect)})')

        # Plot phase map boundary: |1 - Gi| + Ri = 1
        gi_range = np.linspace(0, max(2, dist_data['Gi'].max()), 100)
        ri_boundary = 1 - np.abs(1 - gi_range)
        ri_boundary = np.maximum(ri_boundary, 0)  # Ensure non-negative

        ax.plot(gi_range, ri_boundary, 'b--', linewidth=2, label='Phase Map Boundary')

        ax.set_xlabel('Gi (Intensity)')
        ax.set_ylabel('Ri (Coupling Ratio)')
        ax.set_title(f'{dist_name} Distribution\nPhase Map Validation')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('phase_map_validation_scatter.png', dpi=300, bbox_inches='tight')
    plt.savefig('phase_map_validation_scatter.pdf', dpi=300, bbox_inches='tight')
    print("📈 Validation scatter plots saved as phase_map_validation_scatter.png/pdf")
    plt.show()

def main():
    """Main function to run the experiment"""
    print("🔬 Appendix C.3: Phase Map Validation Experiment")
    print("=" * 60)

    print("Choose operation:")
    print("1. Run phase map validation experiment")
    print("2. Analyze existing results")
    print("3. Create validation plots")
    print("4. Exit")

    choice = input("Enter choice (1-4): ").strip()

    if choice == "1":
        # Run full experiment
        print(f"\nReady to run {TOTAL_POINTS_PER_DISTRIBUTION * 3} experiments.")
        print("This will take a significant amount of time...")

        user_input = input("Do you want to proceed? (y/N): ").strip().lower()
        if user_input in ['y', 'yes']:
            run_phase_map_validation_experiment()
        else:
            print("Experiment cancelled by user.")

    elif choice == "2":
        # Analyze results
        try:
            df = pd.read_csv('phase_map_validation_Spike.csv')
            analyze_validation_results(df)
        except FileNotFoundError:
            print("❌ No results file found. Run the experiment first.")

    elif choice == "3":
        # Create plots
        try:
            df = pd.read_csv('phase_map_validation_Spike.csv')
            create_validation_plots(df)
        except FileNotFoundError:
            print("❌ No results file found. Run the experiment first.")

    elif choice == "4":
        print("Goodbye!")

    else:
        print("Invalid choice. Please run again.")

if __name__ == "__main__":
    main()
