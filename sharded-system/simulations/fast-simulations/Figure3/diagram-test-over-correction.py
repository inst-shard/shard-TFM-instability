#!/usr/bin/env python3
"""
Delay Distribution Shape Experiment
Compares three different delay distributions:
1. Spike at max delay: [0, 0, 0, 0, 1]
2. Uniform distribution: [0.2, 0.2, 0.2, 0.2, 0.2]  
3. Bimodal distribution: [0.4, 0, 0.2, 0, 0.4]
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

# --- Styling ---
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#4472C4', '#E36C09', '#C5504B']  # Blue, Orange, Red
LINE_STYLES = ['-', '--', ':']  # Solid, Dashed, Dotted
LABELS = ['Spike', 'Uniform', 'Bimodal']
TARGET_COLOR = '#2F4F4F'  # Dark Gray

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
        with open('config.yml', 'r') as f:
            content = f.read()
        with open('config_backup.yml', 'w') as f:
            f.write(content)
        print("✅ Created config backup")
        return True
    except Exception as e:
        print(f"❌ Failed to backup config: {e}")
        return False

def restore_config():
    """Restore config from backup"""
    try:
        with open('config_backup.yml', 'r') as f:
            content = f.read()
        with open('config.yml', 'w') as f:
            f.write(content)
        print("✅ Restored original config")
        return True
    except Exception as e:
        print(f"❌ Failed to restore config: {e}")
        return False

def update_config_weights(weights):
    """Update only the delay weights in config.yml while preserving format"""
    try:
        # Read the original file
        with open('config.yml', 'r') as f:
            content = f.read()
        
        # Use regex to find and replace only the weights line
        import re
        
        # Pattern to match the weights line in the delay section
        pattern = r'(delay:\s*\n(?:[^\n]*\n)*?\s*weights:\s*)\[[^\]]*\]'
        replacement = r'\1' + str(weights).replace(' ', '')
        
        new_content = re.sub(pattern, replacement, content)
        
        if new_content != content:
            with open('config.yml', 'w') as f:
                f.write(new_content)
            print(f"✅ Updated config with weights: {weights}")
            return True
        else:
            print("❌ Could not find weights line to update")
            return False
            
    except Exception as e:
        print(f"❌ Failed to update config: {e}")
        return False

def run_simulation():
    """Run the Go simulation"""
    try:
        print("🚀 Running simulation...")
        result = subprocess.run(['go', 'run', '../../../main.go'], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Simulation completed successfully")
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

def parse_log_data(log_file, g_max_values):
    """Parse simulation log and extract Shard 0 load data"""
    data = []
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('╔') or line.startswith('║') or line.startswith('╚'):
                continue
            
            if '│' in line and any(char.isdigit() for char in line.split('│')[1]):
                parts = [part.strip() for part in line.split('│')]
                if len(parts) >= 4:
                    try:
                        step = int(parts[1])
                        
                        # Parse Shard 0 data
                        shard0_data = parts[2].strip()
                        if '/' in shard0_data:
                            components = shard0_data.split('/')
                            if len(components) >= 3:
                                fee = float(components[0].strip())
                                load_str = components[1].strip()
                                
                                # Parse load (handle 'k' for thousands)
                                if 'k' in load_str:
                                    load_gas = float(load_str.replace('k', '')) * 1000
                                else:
                                    load_gas = float(load_str)
                                
                                # Convert to load ratio
                                shard_g_max = g_max_values.get(0, 1600000)
                                load_ratio = load_gas / shard_g_max
                                
                                data.append({
                                    'Step': step,
                                    'Fee': fee,
                                    'Load': load_ratio
                                })
                    except (ValueError, IndexError):
                        continue
        
        df = pd.DataFrame(data)
        print(f"✅ Parsed {len(df)} data points")
        return df
    
    except Exception as e:
        print(f"❌ Error parsing log: {e}")
        return None

def analyze_price_dynamics(df, experiment_label, delay_weights):
    """Analyze basic price dynamics for the experiment"""
    if df is None or len(df) == 0:
        return
    
    print(f"\n📈 Basic Analysis for {experiment_label}")
    print("=" * 60)
    
    # Basic statistics
    fee_mean = df['Fee'].mean()
    fee_std = df['Fee'].std()
    fee_min = df['Fee'].min()
    fee_max = df['Fee'].max()
    
    load_mean = df['Load'].mean()
    load_std = df['Load'].std()
    load_min = df['Load'].min()
    load_max = df['Load'].max()
    
    print(f"Fee Statistics:")
    print(f"   Mean: {fee_mean:.6f}")
    print(f"   Std: {fee_std:.6f}")
    print(f"   Range: [{fee_min:.6f}, {fee_max:.6f}]")
    
    print(f"Load Statistics:")
    print(f"   Mean: {load_mean:.6f}")
    print(f"   Std: {load_std:.6f}")
    print(f"   Range: [{load_min:.6f}, {load_max:.6f}]")
    
    print("-" * 60)

def run_experiment():
    """Run all three experiments and collect data"""
    # Load g_max values
    config = load_config()
    if config is None:
        return None
    
    g_max_values = {}
    for shard in config.get('shards', []):
        g_max_values[shard['id']] = shard['g_max']
    
    # Define the three weight distributions
    weight_sets = [
        [0, 0, 0, 0, 1],           # Spike at max delay
        [0.2, 0.2, 0.2, 0.2, 0.2], # Uniform distribution
        [0.4, 0, 0.2, 0, 0.4]      # Bimodal distribution
    ]
    
    experiment_data = []
    
    for i, weights in enumerate(weight_sets):
        print(f"\n🔬 Running Experiment {i+1}: {LABELS[i]}")
        
        # Update config with new weights
        if not update_config_weights(weights):
            continue
        
        # Run simulation
        if not run_simulation():
            continue
        
        # Wait a moment for file to be written
        time.sleep(1)
        
        # Find and parse the latest log
        log_file = find_latest_log()
        if log_file is None:
            print("❌ No log file found")
            continue
        
        df = parse_log_data(log_file, g_max_values)
        if df is None:
            continue
        
        # Filter to include both visualization intervals (100-300 and 4000-4200)
        df_filtered = df[(df['Step'] >= 100) & (df['Step'] <= 5000)].copy()
        
        if len(df_filtered) > 0:
            experiment_data.append({
                'weights': weights,
                'label': LABELS[i],
                'color': COLORS[i],
                'data': df_filtered
            })
            
            # Calculate statistics
            tail_data = df[df['Step'] >= df['Step'].max() * 0.8]  # Last 20% of data
            avg_load = tail_data['Load'].mean()
            std_load = tail_data['Load'].std()
            print(f"📊 Average Load: {avg_load:.6f}, Std Dev: {std_load:.6f}")
            
            # Analyze price dynamics
            analyze_price_dynamics(df, LABELS[i], weights)
        
        print(f"✅ Experiment {i+1} completed")
    
    return experiment_data

def create_comparison_plot(experiment_data):
    """Create a comparison plot of all three experiments with two subplots"""
    if not experiment_data:
        print("❌ No experiment data to plot")
        return
    
    # Define the two time intervals to plot
    intervals = [(100, 300), (4800, 5000)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 9), sharey=True)
    
    for i, ax in enumerate(axes):
        start, end = intervals[i]
        ax.set_title(f'Blocks {start}-{end}', fontsize=30, fontweight='bold')
        
        # Plot each experiment's data for this interval
        for j, exp in enumerate(experiment_data):
            # Filter data for the current interval
            interval_data = exp['data'][(exp['data']['Step'] >= start) & (exp['data']['Step'] <= end)]
            
            if len(interval_data) > 0:
                ax.plot(
                    interval_data['Step'], 
                    interval_data['Load'],
                    color=exp['color'],
                    linestyle=LINE_STYLES[j],
                    linewidth=2,
                    alpha=1.0,
                    label=exp['label']
                )
        
        # Add target load line
        ax.axhline(y=0.5, color='black', linestyle='-.', linewidth=3, alpha=1.0, label='Target Load')
        
        # Styling (matching visualize-new.py style)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Block Number', fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=26)
        ax.set_ylim(0, 1.1)
        
        # Set custom ticks
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        
        if i == 0:  # First subplot (100-300)
            ax.set_xticks([100, 200, 300])
        else:  # Second subplot (4800-5000)
            ax.set_xticks([4800, 4900, 5000])
    
    # Set shared Y-label
    axes[0].set_ylabel('Block Load', fontsize=28)
    
    # Create a single, unified legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), 
               ncol=4, fontsize=30, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])  # Adjust layout to make space for legend
    
    # Save the plot
    output_filename = 'delay_distribution_comparison.pdf'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"💾 Comparison plot saved as {output_filename}")
    
    plt.show()

def main():
    """Main function"""
    print("🔬 Starting Delay Distribution Shape Experiment")
    print("This will run 3 simulations with different delay distributions:")
    for i, label in enumerate(LABELS):
        print(f"  {i+1}. {label}")
    
    # Backup original config
    if not backup_config():
        print("❌ Failed to backup config, aborting")
        return
    
    try:
        # Run all experiments
        experiment_data = run_experiment()
        
        if experiment_data:
            print(f"\n✅ Successfully completed {len(experiment_data)} experiments")
            create_comparison_plot(experiment_data)
        else:
            print("❌ No experiments completed successfully")
    
    finally:
        # Always restore original config
        restore_config()
        print("🔄 Original config.yml restored")

if __name__ == '__main__':
    main()
