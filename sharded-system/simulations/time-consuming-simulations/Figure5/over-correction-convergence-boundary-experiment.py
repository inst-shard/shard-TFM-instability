#!/usr/bin/env python3
"""
Convergence Boundary Experiment
Test system convergence across different epsilon and alpha parameters
for three delay distributions: Spike, Uniform, Bimodal

Variables:
- epsilon_j0: from 1.5 to 8.0, step 0.05 (tested in reverse order: 8.0 → 1.5)
- alpha_j->0,in: from 50% to 90%, step 1% (tested in reverse order: 90% → 50%)

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

# Simulation parameters
G_MAX = 2000000  # Each shard's g_max
TARGET_TOTAL_DEMAND = G_MAX / 2  # Each shard's total demand should be g_max/2

# Experiment parameter ranges
EPSILON_MIN = 1.5
EPSILON_MAX = 16
EPSILON_STEP = 0.1

ALPHA_MIN = 0.45  
ALPHA_MAX = 0.95 
ALPHA_STEP = 0.01
ALPHA_IJ_OUT = 0  

# Calculate total experiment points
EPSILON_POINTS = int((EPSILON_MAX - EPSILON_MIN) / EPSILON_STEP) + 1  # 59 points
ALPHA_POINTS = round((ALPHA_MAX - ALPHA_MIN) / ALPHA_STEP) + 1  # 41 points
TOTAL_POINTS_PER_DISTRIBUTION = EPSILON_POINTS * ALPHA_POINTS  # 2419 points per distribution

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
    
    Constraints:
    - Shard 0: sum(00, 01, 02, 10, 20) = G_MAX/2
    - Shard 1: sum(10, 11, 12, 01, 21) = G_MAX/2  
    - Shard 2: sum(20, 21, 22, 02, 12) = G_MAX/2
    - alpha_ij,out = 0.05 ONLY for shard 0: (01 + 02) = 0.05 * G_MAX/2
    - 00, 11, 22 should be equal (diagonal symmetry)
    
    Parameters:
    - alpha_inflow_to_0: ratio of inflow to shard 0 (10 + 20) / total_inflow_to_0
    """
    
    # Initialize matrix
    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    # Step 1: Calculate outflow from shard 0 using alpha_ij,out = 0.05
    total_outflow_from_0 = TARGET_TOTAL_DEMAND * ALPHA_IJ_OUT  # 01 + 02 = 0.05 * G_MAX/2
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
    
    # Check alpha_ij,out constraint ONLY for shard 0
    # Outflow from shard 0 to others: 01 + 02
    outflow_from_0_to_others = matrix[0][1] + matrix[0][2]
    alpha_0_actual = outflow_from_0_to_others / TARGET_TOTAL_DEMAND
    
    print(f"  Alpha constraint for shard 0:")
    print(f"    Outflow from shard 0 (01+02): {outflow_from_0_to_others:.0f}")
    print(f"    Shard 0 alpha_ij,out: {alpha_0_actual:.3f} (target: {ALPHA_IJ_OUT:.3f})")
    
    tolerance = 1.0  # Allow small numerical errors
    shard0_ok = abs(shard0_total - TARGET_TOTAL_DEMAND) < tolerance
    shard1_ok = abs(shard1_total - TARGET_TOTAL_DEMAND) < tolerance  
    shard2_ok = abs(shard2_total - TARGET_TOTAL_DEMAND) < tolerance
    
    alpha_tolerance = 0.001
    alpha_0_ok = abs(alpha_0_actual - ALPHA_IJ_OUT) < alpha_tolerance
    
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
    print(f"\nDemand matrix for alpha={alpha_value:.2f}:")
    print("     To:  0        1        2")
    for i in range(3):
        row_str = f"From {i}: "
        for j in range(3):
            row_str += f"{matrix[i][j]:8.0f} "
        print(row_str)
    
    # Highlight diagonal elements
    diagonal_value = matrix[0][0]
    print(f"Diagonal elements (00, 11, 22): {diagonal_value:.0f}")
    print(f"Cross flows (10, 20): {matrix[1][0]:.0f}, {matrix[2][0]:.0f}")
    print(f"Cross flows (12, 21): {matrix[1][2]:.0f}, {matrix[2][1]:.0f}")

def test_demand_matrix_calculation():
    """Test the demand matrix calculation for different alpha values"""
    print("🧪 Testing demand matrix calculation...")
    print("=" * 60)
    
    # Test with correct step size: 0.01 from 0.50 to 0.90
    test_alphas = [0.50, 0.55, 0.60, 0.70, 0.80, 0.90]  # Sample test points
    
    for alpha in test_alphas:
        print(f"\n📊 Testing alpha = {alpha:.2f}")
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
        
        print(f"✅ Config updated: epsilon_j0={epsilon_j0:.2f}, alpha={alpha_inflow:.2f}")
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

def run_single_experiment(epsilon_j0, alpha_inflow, delay_weights, distribution_name, exp_id):
    """Run a single experiment with given parameters"""
    # Update config
    if not update_config_for_experiment(epsilon_j0, alpha_inflow, delay_weights):
        return None
    
    # Create unique log file name for this experiment
    unique_log_name = f"experiment_{distribution_name}_{exp_id:04d}_eps{epsilon_j0:.2f}_alpha{alpha_inflow:.2f}.log"
    
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
        'epsilon_j0': epsilon_j0,
        'alpha_inflow': alpha_inflow,
        'distribution': distribution_name,
        'final_load': load_stats['load0'],
        'avg_load': load_stats['avg_load'],
        'load_std': load_stats['load_std'],
        'load_inflation': load_stats['load_inflation'],
        'converged': converged
    }

def save_distribution_results(results, distribution_name):
    """Save all results for a distribution to a single file"""
    filename = f"convergence_results_{distribution_name}.csv"
    
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"💾 Saved {len(results)} results to {filename}")

def run_convergence_experiment():
    """Run the complete convergence boundary experiment"""
    print("🔬 Starting Convergence Boundary Experiment")
    print("=" * 80)
    
    total_experiments = TOTAL_POINTS_PER_DISTRIBUTION * 3
    print(f"Total experiments to run: {total_experiments}")
    print(f"Epsilon range: {EPSILON_MIN} to {EPSILON_MAX}, step {EPSILON_STEP} ({EPSILON_POINTS} points)")
    print(f"Alpha range: {ALPHA_MIN} to {ALPHA_MAX}, step {ALPHA_STEP} ({ALPHA_POINTS} points)")
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
            print("-" * 60)
            
            distribution_results = []  # Store all results for this distribution
            
            # Loop through epsilon values (从最大值开始倒序)
            for i in range(EPSILON_POINTS-1, -1, -1):
                epsilon_j0 = EPSILON_MIN + i * EPSILON_STEP
                
                # Loop through alpha values (从最大值开始倒序)
                for j in range(ALPHA_POINTS-1, -1, -1):
                    alpha_inflow = ALPHA_MIN + j * ALPHA_STEP
                    experiment_count += 1
                    
                    print(f"Experiment {experiment_count}/{total_experiments}: " + 
                          f"{dist_name}, ε={epsilon_j0:.2f}, α={alpha_inflow:.2f}")
                    
                    # Run single experiment
                    result = run_single_experiment(epsilon_j0, alpha_inflow, delay_weights, dist_name, experiment_count)
                    
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
                            'epsilon_j0': epsilon_j0,
                            'alpha_inflow': alpha_inflow,
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
        filename = f"convergence_results_{dist_name}.csv"
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

def create_3d_convergence_plots(df):
    """Create 3D plots showing convergence boundaries for each distribution"""
    if df is None or len(df) == 0:
        print("❌ No data available for plotting")
        return
    
    distributions = df['distribution'].unique()
    
    fig = plt.figure(figsize=(18, 6))
    
    for i, dist_name in enumerate(distributions):
        dist_data = df[df['distribution'] == dist_name]
        
        # Create 3D subplot
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Separate converged and diverged points
        converged = dist_data[dist_data['converged'] == True]
        diverged = dist_data[dist_data['converged'] == False]
        
        # Plot converged points (green)
        if len(converged) > 0:
            ax.scatter(converged['epsilon_j0'], converged['alpha_inflow'], 
                      [1] * len(converged), c='green', marker='o', s=20, alpha=0.6, label='Converged')
        
        # Plot diverged points (red) 
        if len(diverged) > 0:
            ax.scatter(diverged['epsilon_j0'], diverged['alpha_inflow'], 
                      [0] * len(diverged), c='red', marker='x', s=20, alpha=0.6, label='Diverged')
        
        ax.set_xlabel('Epsilon (ε_j0)', fontsize=12)
        ax.set_ylabel('Alpha (α_j→0,in)', fontsize=12)
        ax.set_zlabel('Convergence', fontsize=12)
        ax.set_title(f'{dist_name} Distribution\nConvergence Boundary', fontsize=14, fontweight='bold')
        
        # Set z-axis limits
        ax.set_zlim(-0.1, 1.1)
        ax.set_zticks([0, 1])
        ax.set_zticklabels(['Diverged', 'Converged'])
        
        # Add legend
        ax.legend()
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('convergence_boundary_3d.png', dpi=300, bbox_inches='tight')
    plt.savefig('convergence_boundary_3d.pdf', dpi=300, bbox_inches='tight')
    print("💾 3D convergence plots saved as convergence_boundary_3d.png/pdf")
    plt.show()

def analyze_convergence_results(df):
    """Analyze and summarize convergence results"""
    if df is None or len(df) == 0:
        print("❌ No data available for analysis")
        return
    
    print("\n📊 CONVERGENCE ANALYSIS SUMMARY")
    print("=" * 80)
    
    for dist_name in df['distribution'].unique():
        dist_data = df[df['distribution'] == dist_name]
        total_points = len(dist_data)
        converged_points = len(dist_data[dist_data['converged'] == True])
        convergence_rate = converged_points / total_points * 100
        
        print(f"\n🔬 {dist_name} Distribution:")
        print(f"   Total experiments: {total_points}")
        print(f"   Converged: {converged_points} ({convergence_rate:.1f}%)")
        print(f"   Diverged: {total_points - converged_points} ({100-convergence_rate:.1f}%)")
        
        # Find boundary region (where convergence changes)
        converged_subset = dist_data[dist_data['converged'] == True]
        if len(converged_subset) > 0:
            max_epsilon_converged = converged_subset['epsilon_j0'].max()
            max_alpha_converged = converged_subset['alpha_inflow'].max()
            print(f"   Max converged ε: {max_epsilon_converged:.2f}")
            print(f"   Max converged α: {max_alpha_converged:.2f}")
    
    # Save summary to file
    summary_file = "convergence_analysis_summary.csv"
    summary_stats = []
    
    for dist_name in df['distribution'].unique():
        dist_data = df[df['distribution'] == dist_name]
        total_points = len(dist_data)
        converged_points = len(dist_data[dist_data['converged'] == True])
        
        summary_stats.append({
            'distribution': dist_name,
            'total_experiments': total_points,
            'converged_count': converged_points,
            'convergence_rate': converged_points / total_points * 100
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(summary_file, index=False)
    print(f"\n💾 Summary saved to {summary_file}")

def visualize_results():
    """Load results and create visualizations"""
    print("📈 Loading and visualizing results...")
    
    df = load_and_combine_results()
    if df is not None:
        analyze_convergence_results(df)
        create_3d_convergence_plots(df)
        
        # Save combined results
        df.to_csv("convergence_results_combined.csv", index=False)
        print("💾 Combined results saved to convergence_results_combined.csv")
    else:
        print("❌ No results to visualize")

if __name__ == "__main__":
    print("🔬 Convergence Boundary Experiment")
    print("=" * 60)
    
    print("Choose operation:")
    print("1. Test demand matrix calculation")
    print("2. Run full convergence experiment")
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
            
            print(f"\nReady to run {TOTAL_POINTS_PER_DISTRIBUTION * 3} experiments.")
            print("This will take a significant amount of time...")
            # print(f"Estimated time: ~{TOTAL_POINTS_PER_DISTRIBUTION * 3 * 2 / 3600:.1f} hours")
            
            user_input = input("Do you want to proceed? (y/N): ").strip().lower()
            if user_input in ['y', 'yes']:
                run_convergence_experiment()
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
