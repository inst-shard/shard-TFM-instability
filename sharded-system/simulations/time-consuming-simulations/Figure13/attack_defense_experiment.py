#!/usr/bin/env python3
"""
Attack > Defense Experiment
Test system behavior when attack > defense across different epsilon and alpha parameters
for three delay distributions: Spike, Uniform, Bimodal

Variables:
- epsilon_0j: 
  * Spike: from 4.4 to 8.0, step 0.1
  * Uniform: from 8.0 to 14.0, step 0.1  
  * Bimodal: from 8.0 to 14.0, step 0.1
- alpha_0j,out: from 50% to 90%, step 1% (tested in reverse order: 90% → 50%)
- alpha_ji,in: fixed at 0.1 (10%)

Convergence criterion: load == 0.5 at step 5000
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

# Delay distribution configurations
DELAY_DISTRIBUTIONS = {
    'Spike': [0, 0, 0, 0, 1],
    'Uniform': [0.2, 0.2, 0.2, 0.2, 0.2],
    'Bimodal': [0.4, 0, 0.2, 0, 0.4]
}

# Epsilon ranges for different distributions
EPSILON_RANGES = {
    'Spike': {'min': 1.5, 'max': 16},
    'Uniform': {'min': 1.5, 'max': 16},
    'Bimodal': {'min': 1.5, 'max': 16}
}

# Simulation parameters
G_MAX = 2000000  # Each shard's g_max
TARGET_TOTAL_DEMAND = G_MAX / 2  # Each shard's total demand should be g_max/2

# Experiment parameter ranges
EPSILON_STEP = 0.1

ALPHA_MIN = 0.45  # 45%
ALPHA_MAX = 0.95  # 95%
ALPHA_STEP = 0.01  # 1% step size
ALPHA_JI_IN = 0  # Fixed alpha_ji,in = 5%

# Calculate total experiment points for each distribution
# Fix floating point precision issue by using round()
ALPHA_POINTS = round((ALPHA_MAX - ALPHA_MIN) / ALPHA_STEP) + 1  # Should be 51 points

def calculate_epsilon_points(dist_name):
    """Calculate number of epsilon points for a given distribution"""
    eps_range = EPSILON_RANGES[dist_name]
    return int((eps_range['max'] - eps_range['min']) / EPSILON_STEP) + 1

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

def calculate_demand_matrix(alpha_outflow_from_0):
    """
    Calculate base_demand_matrix based on alpha_0j,out (outflow ratio from shard 0)
    
    Constraints:
    - Shard 0: sum(00, 01, 02, 10, 20) = G_MAX/2
    - Shard 1: sum(10, 11, 12, 01, 21) = G_MAX/2  
    - Shard 2: sum(20, 21, 22, 02, 12) = G_MAX/2
    - alpha_ji,in = 0.1 ONLY for shard 0: (10 + 20) = 0.1 * G_MAX/2
    - All shards have equal local flows: 00 = 11 = 22
    
    Parameters:
    - alpha_outflow_from_0: ratio of outflow from shard 0 (01 + 02) / total_demand_shard_0
    """
    
    # Initialize matrix
    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    # Step 1: Calculate outflow from shard 0 (01 + 02)
    total_outflow_from_0 = TARGET_TOTAL_DEMAND * alpha_outflow_from_0
    matrix[0][1] = total_outflow_from_0 / 2  # 01
    matrix[0][2] = total_outflow_from_0 / 2  # 02
    
    # Step 2: Calculate inflow to shard 0 using alpha_ji,in = 0.1
    total_inflow_to_0 = TARGET_TOTAL_DEMAND * ALPHA_JI_IN  # 10 + 20 = 0.1 * G_MAX/2
    matrix[1][0] = total_inflow_to_0 / 2  # 10
    matrix[2][0] = total_inflow_to_0 / 2  # 20
    
    # Step 3: Calculate 00 to satisfy shard 0 total demand constraint
    matrix[0][0] = TARGET_TOTAL_DEMAND - matrix[0][1] - matrix[0][2] - matrix[1][0] - matrix[2][0]  # 00
    
    # Step 4: Set all diagonal elements equal to 00
    local_flow = matrix[0][0]
    matrix[1][1] = local_flow  # 11 = 00
    matrix[2][2] = local_flow  # 22 = 00
    
    # Step 5: Calculate remaining cross-shard flows
    # For shard 1: 10 + 11 + 12 + 01 + 21 = G_MAX/2
    # We know: 10, 11, 01
    # Remaining for 12 + 21
    remaining_cross_1 = TARGET_TOTAL_DEMAND - matrix[1][0] - matrix[1][1] - matrix[0][1]
    
    # For shard 2: 20 + 21 + 22 + 02 + 12 = G_MAX/2
    # We know: 20, 22, 02
    # Remaining for 21 + 12
    remaining_cross_2 = TARGET_TOTAL_DEMAND - matrix[2][0] - matrix[2][2] - matrix[0][2]
    
    # Since we have two equations:
    # 12 + 21 = remaining_cross_1
    # 21 + 12 = remaining_cross_2
    # And they should be equal due to symmetry, we use the average
    avg_remaining_cross = (remaining_cross_1 + remaining_cross_2) / 2
    
    # Split evenly: 12 = 21
    matrix[1][2] = avg_remaining_cross / 2  # 12
    matrix[2][1] = avg_remaining_cross / 2  # 21
    
    return matrix

def verify_demand_matrix(matrix):
    """Verify that the demand matrix satisfies all constraints"""
    
    # Check shard 0: 00 + 01 + 02 + 10 + 20 = G_MAX/2
    shard0_total = matrix[0][0] + matrix[0][1] + matrix[0][2] + matrix[1][0] + matrix[2][0]
    
    # Check shard 1: 10 + 11 + 12 + 01 + 21 = G_MAX/2
    shard1_total = matrix[1][0] + matrix[1][1] + matrix[1][2] + matrix[0][1] + matrix[2][1]
    
    # Check shard 2: 20 + 21 + 22 + 02 + 12 = G_MAX/2
    shard2_total = matrix[2][0] + matrix[2][1] + matrix[2][2] + matrix[0][2] + matrix[1][2]
    
    print(f"Demand matrix verification:")
    print(f"  Shard 0 total: {shard0_total:.0f} (target: {TARGET_TOTAL_DEMAND:.0f})")
    print(f"  Shard 1 total: {shard1_total:.0f} (target: {TARGET_TOTAL_DEMAND:.0f})")
    print(f"  Shard 2 total: {shard2_total:.0f} (target: {TARGET_TOTAL_DEMAND:.0f})")
    
    # Check alpha_ji,in constraint ONLY for shard 0
    # Inflow to shard 0 from others: 10 + 20
    inflow_to_0_from_others = matrix[1][0] + matrix[2][0]
    alpha_0_actual = inflow_to_0_from_others / TARGET_TOTAL_DEMAND
    
    print(f"  Alpha constraint for shard 0:")
    print(f"    Inflow to shard 0 (10+20): {inflow_to_0_from_others:.0f}")
    print(f"    Shard 0 alpha_ji,in: {alpha_0_actual:.3f} (target: {ALPHA_JI_IN:.3f})")
    
    tolerance = 1.0  # Allow small numerical errors
    shard0_ok = abs(shard0_total - TARGET_TOTAL_DEMAND) < tolerance
    shard1_ok = abs(shard1_total - TARGET_TOTAL_DEMAND) < tolerance  
    shard2_ok = abs(shard2_total - TARGET_TOTAL_DEMAND) < tolerance
    
    alpha_tolerance = 0.001
    alpha_0_ok = abs(alpha_0_actual - ALPHA_JI_IN) < alpha_tolerance
    
    all_positive = all(matrix[i][j] >= 0 for i in range(3) for j in range(3))
    
    is_valid = shard0_ok and shard1_ok and shard2_ok and alpha_0_ok and all_positive
    
    print(f"  Constraints satisfied: {is_valid}")
    print(f"    Shard totals OK: {shard0_ok}, {shard1_ok}, {shard2_ok}")
    print(f"    Alpha constraint (shard 0) OK: {alpha_0_ok}")
    print(f"    All positive: {all_positive}")
    
    if not all_positive:
        print("  ❌ Negative values found:")
        for i in range(3):
            for j in range(3):
                if matrix[i][j] < 0:
                    print(f"    matrix[{i}][{j}] = {matrix[i][j]:.2f}")
    
    return is_valid

def print_demand_matrix(matrix, alpha_value):
    """Print the demand matrix in a readable format"""
    print(f"\nDemand matrix for alpha_0j,out={alpha_value:.2f}:")
    print("     To:  0        1        2")
    for i in range(3):
        row_str = f"From {i}: "
        for j in range(3):
            row_str += f"{matrix[i][j]:8.0f} "
        print(row_str)
    
    # Highlight key flows
    print(f"Outflows from shard 0 (01, 02): {matrix[0][1]:.0f}, {matrix[0][2]:.0f}")
    print(f"Inflows to shard 0 (10, 20): {matrix[1][0]:.0f}, {matrix[2][0]:.0f}")
    print(f"Cross flows (12, 21): {matrix[1][2]:.0f}, {matrix[2][1]:.0f}")

def test_demand_matrix_calculation():
    """Test the demand matrix calculation for different alpha values"""
    print("🧪 Testing demand matrix calculation...")
    print("=" * 60)
    
    # Test with sample alpha values
    test_alphas = [0.50, 0.55, 0.60, 0.70, 0.80, 0.90]  # Sample test points
    
    for alpha in test_alphas:
        print(f"\n📊 Testing alpha_0j,out = {alpha:.2f}")
        print("-" * 40)
        
        matrix = calculate_demand_matrix(alpha)
        print_demand_matrix(matrix, alpha)
        is_valid = verify_demand_matrix(matrix)
        
        if not is_valid:
            print(f"❌ Invalid matrix for alpha = {alpha}")
            return False
        else:
            print(f"✅ Valid matrix for alpha = {alpha}")
    
    print(f"\n📏 Full range validation:")
    print(f"   Alpha range: 0.50 to 0.90")
    print(f"   Step size: 0.01")
    print(f"   Total points: {int((0.90 - 0.50) / 0.01) + 1} = 41 points")
    print(f"   Fixed alpha_ji,in: {ALPHA_JI_IN}")
    
    # Test edge cases
    edge_alphas = [0.50, 0.51, 0.89, 0.90]
    print(f"\n🧪 Testing edge cases...")
    for alpha in edge_alphas:
        matrix = calculate_demand_matrix(alpha)
        is_valid = verify_demand_matrix(matrix)
        if not is_valid:
            print(f"❌ Edge case failed for alpha = {alpha}")
            return False
        print(f"✅ Edge case passed for alpha = {alpha:.2f}")
    
    print("\n🎉 All test cases passed!")
    return True

def update_config_for_experiment(epsilon_0j, alpha_outflow, delay_weights):
    """Update config.yml for the experiment parameters"""
    try:
        config = load_config()
        if config is None:
            return False
        
        # Update epsilon matrix - change epsilon for shard 0 outflows (row 0)
        config['demand']['epsilon_matrix'][0][1] = epsilon_0j  # epsilon_01
        config['demand']['epsilon_matrix'][0][2] = epsilon_0j  # epsilon_02
        
        # Calculate and update demand matrix
        demand_matrix = calculate_demand_matrix(alpha_outflow)
        config['demand']['base_demand_matrix'] = demand_matrix
        
        # Update delay weights
        config['delay']['weights'] = delay_weights
        
        # Write updated config
        with open('config.yml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Config updated: epsilon_0j={epsilon_0j:.2f}, alpha_out={alpha_outflow:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update config: {e}")
        return False

def run_simulation():
    """Run the Go simulation"""
    try:
        result = subprocess.run(['go', 'run', '../../../main.go'], 
                              capture_output=True, text=True, timeout=60)
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
    """Find the most recent simulation log file"""
    enhanced_logs = glob.glob("enhanced_simulation_analysis_*.log")
    if enhanced_logs:
        latest_log = max(enhanced_logs, key=os.path.getctime)
        return latest_log
    return None

def parse_final_load(log_file):
    """Parse simulation log and extract shard0 load statistics from all CSV data"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        print(f"  📄 Parsing log file: {log_file}")
        
        # Find CSV data section
        data_start_idx = content.find('DATA_START')
        if data_start_idx == -1:
            print(f"  ❌ No DATA_START found in {log_file}")
            return None
            
        # Extract CSV data
        csv_section = content[data_start_idx:]
        lines = csv_section.split('\n')
        
        csv_data = []
        header_found = False
        for line in lines:
            line = line.strip()
            if line.startswith('Step,Shard0_Fee'):
                header_found = True
                csv_data.append(line)
            elif header_found and ',' in line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) >= 7:  # Step,Shard0_Fee,Shard1_Fee,Shard2_Fee,Shard0_Load,Shard1_Load,Shard2_Load
                    try:
                        int(parts[0])  # Verify step number
                        csv_data.append(line)
                    except ValueError:
                        break  # Stop at invalid data
        
        if len(csv_data) <= 1:
            print(f"  ❌ No valid CSV data found")
            return None
            
        # Parse CSV data using pandas
        import pandas as pd
        from io import StringIO
        
        csv_text = '\n'.join(csv_data)
        df = pd.read_csv(StringIO(csv_text))
        
        # Extract shard0 load data only
        shard0_loads = df['Shard0_Load']
        
        # Calculate statistics for shard0 only
        final_load = shard0_loads.iloc[-1]  # Last step load
        avg_load = shard0_loads.mean()      # Average load across all steps
        load_std = shard0_loads.std()       # Standard deviation of shard0 loads
        target_load = 0.5  # Target load is 50%
        load_inflation = (avg_load - target_load) / target_load  # Load inflation relative to target
        
        print(f"  ✅ Parsed {len(shard0_loads)} steps of shard0 data")
        print(f"  📊 Shard0 stats: final={final_load:.6f}, avg={avg_load:.6f}, std={load_std:.6f}")
        
        return {
            'load0': final_load,      # Final step load for shard0
            'avg_load': avg_load,     # Average shard0 load across all steps
            'load_std': load_std,     # Standard deviation of shard0 loads
            'load_inflation': load_inflation  # Load inflation relative to target
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
    """Remove all remaining log files to save memory"""
    try:
        # Clean enhanced simulation logs
        enhanced_logs = glob.glob("enhanced_simulation_analysis_*.log")
        experiment_logs = glob.glob("experiment_*.log")
        all_logs = enhanced_logs + experiment_logs
        
        if all_logs:
            for log_file in all_logs:
                os.remove(log_file)
            print(f"  🧹 Cleaned {len(all_logs)} remaining log files")
        return True
    except Exception as e:
        print(f"⚠️  Warning: Could not clean log files: {e}")
        return False

def check_convergence(load_stats, tolerance=0.00001):
    """Check if the system converged to target load (0.5)"""
    if load_stats is None or not isinstance(load_stats, dict):
        return False
    return abs(load_stats['load0'] - 0.5) < tolerance

def run_single_experiment(epsilon_0j, alpha_outflow, delay_weights, distribution_name, exp_id):
    """Run a single experiment with given parameters"""
    # Update config
    if not update_config_for_experiment(epsilon_0j, alpha_outflow, delay_weights):
        return None
    
    # Create unique log file name for this experiment
    unique_log_name = f"experiment_{distribution_name}_{exp_id:04d}_eps{epsilon_0j:.2f}_alpha{alpha_outflow:.2f}.log"
    
    # Run simulation
    if not run_simulation():
        return None
    
    # Find and rename the latest log to unique name
    latest_log = find_latest_log()
    if latest_log is None:
        print(f"  ❌ No log file found for experiment {exp_id}")
        return None
    
    try:
        # Rename log file to unique name
        os.rename(latest_log, unique_log_name)
        print(f"  📝 Renamed log to: {unique_log_name}")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not rename log file: {e}")
        unique_log_name = latest_log
    
    # Parse results
    load_stats = parse_final_load(unique_log_name)
    converged = check_convergence(load_stats)
    
    # Clean up the log file immediately after parsing
    try:
        os.remove(unique_log_name)
        print(f"  🗑️  Cleaned up: {unique_log_name}")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not remove log file: {e}")
    
    return {
        'epsilon_0j': epsilon_0j,
        'alpha_outflow': alpha_outflow,
        'distribution': distribution_name,
        'final_load': load_stats['load0'],
        'avg_load': load_stats['avg_load'],
        'load_std': load_stats['load_std'],
        'load_inflation': load_stats['load_inflation'],
        'converged': converged
    }

def save_distribution_results(results, distribution_name):
    """Save all results for a distribution to a single file"""
    filename = f"attack_defense_results_{distribution_name}.csv"
    
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"💾 Saved {len(results)} results to {filename}")

def run_attack_defense_experiment():
    """Run the complete attack > defense experiment"""
    print("🔬 Starting Attack > Defense Experiment")
    print("=" * 80)
    
    # Calculate total experiments
    total_experiments = 0
    for dist_name in DELAY_DISTRIBUTIONS.keys():
        epsilon_points = calculate_epsilon_points(dist_name)
        total_experiments += epsilon_points * ALPHA_POINTS
    
    print(f"Total experiments to run: {total_experiments}")
    print(f"Alpha range: {ALPHA_MIN} to {ALPHA_MAX}, step {ALPHA_STEP} ({ALPHA_POINTS} points)")
    print(f"Fixed alpha_ji,in: {ALPHA_JI_IN}")
    print(f"Epsilon ranges:")
    for dist_name, eps_range in EPSILON_RANGES.items():
        epsilon_points = calculate_epsilon_points(dist_name)
        print(f"  {dist_name}: {eps_range['min']} to {eps_range['max']}, step {EPSILON_STEP} ({epsilon_points} points)")
    print(f"Delay distributions: {list(DELAY_DISTRIBUTIONS.keys())}")
    print(f"Each distribution will be saved to its own CSV file")
    
    # Backup original config
    if not backup_config():
        print("❌ Failed to backup config, aborting")
        return
    
    try:
        experiment_count = 0
        
        # Loop through each delay distribution
        for dist_name, delay_weights in DELAY_DISTRIBUTIONS.items():
            print(f"\n🧪 Testing delay distribution: {dist_name}")
            print(f"Weights: {delay_weights}")
            
            eps_range = EPSILON_RANGES[dist_name]
            epsilon_points = calculate_epsilon_points(dist_name)
            print(f"Epsilon range: {eps_range['min']} to {eps_range['max']} ({epsilon_points} points)")
            print("-" * 60)
            
            distribution_results = []  # Store all results for this distribution
            
            # Loop through epsilon values (从最大值开始倒序)
            for i in range(epsilon_points-1, -1, -1):
                epsilon_0j = eps_range['min'] + i * EPSILON_STEP
                
                # Loop through alpha values (从最大值开始倒序)
                for j in range(ALPHA_POINTS-1, -1, -1):
                    alpha_outflow = ALPHA_MIN + j * ALPHA_STEP
                    experiment_count += 1
                    
                    print(f"Experiment {experiment_count}/{total_experiments}: " + 
                          f"{dist_name}, ε={epsilon_0j:.2f}, α_out={alpha_outflow:.2f}")
                    
                    # Run single experiment
                    result = run_single_experiment(epsilon_0j, alpha_outflow, delay_weights, dist_name, experiment_count)
                    
                    if result is not None:
                        distribution_results.append(result)
                        convergence_status = "✅ Converged" if result['converged'] else "❌ Diverged"
                        final_load_str = f"{result['final_load']:.6f}"
                        avg_load_str = f"{result['avg_load']:.6f}"
                        std_str = f"{result['load_std']:.6f}"
                        inflation_str = f"{result['load_inflation']:.6f}"
                        print(f"  → Load0: {final_load_str}, Avg: {avg_load_str}, Std: {std_str}, Inflation: {inflation_str}, {convergence_status}")
                    else:
                        print(f"  → ❌ Experiment failed")
                        # Add failed experiment record
                        distribution_results.append({
                            'epsilon_0j': epsilon_0j,
                            'alpha_outflow': alpha_outflow,
                            'distribution': dist_name,
                            'final_load': None,
                            'avg_load': None,
                            'load_std': None,
                            'load_inflation': None,
                            'converged': False
                        })
                    
                    # Save results every 10 experiments to prevent data loss
                    if len(distribution_results) % 10 == 0:
                        save_distribution_results(distribution_results, dist_name)
                        print(f"  💾 Saved {len(distribution_results)} results to {dist_name} CSV")
                    
                    # Clean log files every 50 experiments to prevent memory issues
                    if experiment_count % 50 == 0:
                        clean_log_files()
                        print(f"  🧹 Cleaned log files (experiment {experiment_count})")
            
            # Save final results for this distribution
            save_distribution_results(distribution_results, dist_name)
            print(f"✅ Completed {dist_name} distribution: {len(distribution_results)} experiments")
        
        print(f"\n🎉 All experiments completed!")
        print(f"Total experiments run: {experiment_count}")
        print(f"Results saved in 3 CSV files (one per distribution)")
        
    finally:
        # Always restore original config
        restore_config()
        print("🔄 Original config.yml restored")

def load_and_combine_results():
    """Load all distribution result files and combine them"""
    all_results = []
    
    # Find distribution result files
    distribution_files = []
    for dist_name in DELAY_DISTRIBUTIONS.keys():
        filename = f"attack_defense_results_{dist_name}.csv"
        if os.path.exists(filename):
            distribution_files.append(filename)
    
    print(f"Found {len(distribution_files)} distribution result files")
    
    for result_file in distribution_files:
        try:
            df = pd.read_csv(result_file)
            all_results.append(df)
            print(f"Loaded {len(df)} results from {result_file}")
        except Exception as e:
            print(f"❌ Error loading {result_file}: {e}")
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        print(f"✅ Combined {len(combined_df)} total results")
        return combined_df
    else:
        print("❌ No results found")
        return None

def create_2d_plots(df):
    """Create 2D plots showing attack > defense behavior for each distribution"""
    if df is None or len(df) == 0:
        print("❌ No data available for plotting")
        return
    
    distributions = df['distribution'].unique()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, dist_name in enumerate(distributions):
        dist_data = df[df['distribution'] == dist_name]
        
        # Create pivot table for heatmap
        pivot_data = dist_data.pivot_table(
            values='final_load', 
            index='alpha_outflow', 
            columns='epsilon_0j', 
            aggfunc='mean'
        )
        
        # Create heatmap
        im = axes[i].imshow(pivot_data.values, cmap='RdYlBu_r', aspect='auto', origin='lower')
        
        # Set labels and title
        axes[i].set_xlabel('Epsilon (ε_0j)', fontsize=12)
        axes[i].set_ylabel('Alpha (α_0j,out)', fontsize=12)
        axes[i].set_title(f'{dist_name} Distribution\nAttack > Defense', fontsize=14, fontweight='bold')
        
        # Set tick labels
        epsilon_ticks = np.arange(0, len(pivot_data.columns), max(1, len(pivot_data.columns)//10))
        alpha_ticks = np.arange(0, len(pivot_data.index), max(1, len(pivot_data.index)//10))
        
        axes[i].set_xticks(epsilon_ticks)
        axes[i].set_yticks(alpha_ticks)
        
        if len(epsilon_ticks) > 0:
            axes[i].set_xticklabels([f'{pivot_data.columns[t]:.1f}' for t in epsilon_ticks])
        if len(alpha_ticks) > 0:
            axes[i].set_yticklabels([f'{pivot_data.index[t]:.2f}' for t in alpha_ticks])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[i])
        cbar.set_label('Final Load (Shard 0)', fontsize=10)
    
    plt.tight_layout()
    # plt.savefig('attack_defense_2d.png', dpi=300, bbox_inches='tight')
    plt.savefig('attack_defense_2d.pdf', dpi=300, bbox_inches='tight')
    print("💾 2D attack > defense plots saved as attack_defense_2d.png/pdf")
    plt.show()

def analyze_attack_defense_results(df):
    """Analyze and summarize attack > defense results"""
    if df is None or len(df) == 0:
        print("❌ No data available for analysis")
        return
    
    print("\n📊 ATTACK > DEFENSE ANALYSIS SUMMARY")
    print("=" * 80)
    
    for dist_name in df['distribution'].unique():
        dist_data = df[df['distribution'] == dist_name]
        total_points = len(dist_data)
        converged_points = len(dist_data[dist_data['converged'] == True])
        convergence_rate = converged_points / total_points * 100
        
        # Calculate load statistics
        avg_final_load = dist_data['final_load'].mean()
        max_final_load = dist_data['final_load'].max()
        min_final_load = dist_data['final_load'].min()
        
        print(f"\n🔬 {dist_name} Distribution:")
        print(f"   Total experiments: {total_points}")
        print(f"   Converged: {converged_points} ({convergence_rate:.1f}%)")
        print(f"   Diverged: {total_points - converged_points} ({100-convergence_rate:.1f}%)")
        print(f"   Load statistics:")
        print(f"     Average final load: {avg_final_load:.6f}")
        print(f"     Max final load: {max_final_load:.6f}")
        print(f"     Min final load: {min_final_load:.6f}")
        
        # Find extreme cases
        max_load_case = dist_data.loc[dist_data['final_load'].idxmax()]
        min_load_case = dist_data.loc[dist_data['final_load'].idxmin()]
        
        print(f"   Extreme cases:")
        print(f"     Max load: ε={max_load_case['epsilon_0j']:.2f}, α={max_load_case['alpha_outflow']:.2f}")
        print(f"     Min load: ε={min_load_case['epsilon_0j']:.2f}, α={min_load_case['alpha_outflow']:.2f}")
    
    # Save summary to file
    summary_file = "attack_defense_analysis_summary.csv"
    summary_stats = []
    
    for dist_name in df['distribution'].unique():
        dist_data = df[df['distribution'] == dist_name]
        total_points = len(dist_data)
        converged_points = len(dist_data[dist_data['converged'] == True])
        
        summary_stats.append({
            'distribution': dist_name,
            'total_experiments': total_points,
            'converged_count': converged_points,
            'convergence_rate': converged_points / total_points * 100,
            'avg_final_load': dist_data['final_load'].mean(),
            'max_final_load': dist_data['final_load'].max(),
            'min_final_load': dist_data['final_load'].min()
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(summary_file, index=False)
    print(f"\n💾 Summary saved to {summary_file}")

def visualize_results():
    """Load results and create visualizations"""
    print("📈 Loading and visualizing results...")
    
    df = load_and_combine_results()
    if df is not None:
        analyze_attack_defense_results(df)
        create_2d_plots(df)
        
        # Save combined results
        df.to_csv("attack_defense_results_combined.csv", index=False)
        print("💾 Combined results saved to attack_defense_results_combined.csv")
    else:
        print("❌ No results to visualize")

if __name__ == "__main__":
    print("🔬 Attack > Defense Experiment")
    print("=" * 60)
    
    print("Choose operation:")
    print("1. Test demand matrix calculation")
    print("2. Run full attack > defense experiment")
    print("3. Visualize existing results")
    print("4. Exit")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        # Test demand matrix calculation
        if test_demand_matrix_calculation():
            print("\n✅ Demand matrix calculation verified!")
        else:
            print("\n❌ Demand matrix calculation has issues.")
    
    elif choice == "2":
        # Run full experiment
        if test_demand_matrix_calculation():
            print("\n✅ Demand matrix calculation verified!")
            
            # Calculate total experiments
            total_experiments = 0
            for dist_name in DELAY_DISTRIBUTIONS.keys():
                epsilon_points = calculate_epsilon_points(dist_name)
                total_experiments += epsilon_points * ALPHA_POINTS
            
            print(f"\nReady to run {total_experiments} experiments.")
            print("This will take a significant amount of time...")
            print(f"Estimated time: ~{total_experiments * 2 / 3600:.1f} hours")
            
            user_input = input("Do you want to proceed? (y/N): ").strip().lower()
            if user_input in ['y', 'yes']:
                run_attack_defense_experiment()
            else:
                print("Experiment cancelled by user.")
        else:
            print("\n❌ Please fix demand matrix calculation first.")
    
    elif choice == "3":
        # Visualize results
        visualize_results()
    
    elif choice == "4":
        print("Goodbye!")
    
    else:
        print("Invalid choice. Please run again.")
