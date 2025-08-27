#!/usr/bin/env python3
"""
This script automates the process of generating simulation data     print("\n--- 📊 Starting Combined Visualization ---")
    
    # Calculate and print load statistics for each lambda
    print("\n--- 📈 Load Statistics ---")
    for lam in ELASTICITY_VALUES:
        log_file = f"simulation_lambda_{lam}.log"
        try:
            df = pd.read_csv(log_file)
            load_avg = df['Load'].mean()
            load_std = df['Load'].std()
            print(f"λ = {lam:4.1f}: Average Load = {load_avg:.4f}, Load Deviation = {load_std:.4f}")
        except FileNotFoundError:
            print(f"λ = {lam:4.1f}: ⚠️ Log file not found: {log_file}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 8.5), sharey=True) different
elasticity values and then creates a combined visualization to compare their
load dynamics over time.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import subprocess
import os
import time

# --- Configuration ---
ELASTICITY_VALUES = [4.4, 16.5, 25]
CONFIG_FILE = 'config.yml'
SIMULATION_STEPS = 3000  # Ensure simulation runs long enough
INTERVALS = [(0, 50), (2500, 2550)]

# --- Styling ---
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    4.4: '#4472C4',   # Soft Blue (Stable)
    16.5: '#E36C09',  # Soft Orange (Boundary)
    25: '#C5504B'     # Soft Red (Chaotic)
}
LINE_STYLES = {
    4.4: '-',
    16.5: '--',
    25: ':'
}

def generate_simulation_logs():
    """
    Generates simulation logs for different elasticity values by modifying
    the config file and running the Go simulation.
    """
    print("---  Starting Simulation Log Generation ---")
    
    # Backup original config file
    original_config_content = None
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            original_config_content = f.read()
            
    try:
        for lam in ELASTICITY_VALUES:
            log_filename = f"simulation_lambda_{lam}.log"
            print(f"\n Generating log for λ = {lam} -> {log_filename}")
            
            # Load and modify config
            with open(CONFIG_FILE, 'r') as f:
                config = yaml.safe_load(f)
            
            config['demand']['price_elasticity'] = lam
            config['output']['log_file'] = log_filename
            config['simulation']['steps'] = SIMULATION_STEPS
            
            with open(CONFIG_FILE, 'w') as f:
                yaml.dump(config, f)
            
            # Run the Go simulation
            result = subprocess.run(
                ['go', 'run', '../../main.go'],
                capture_output=True, text=True, timeout=60
            )
            
            if result.returncode == 0:
                print(f" Success: {log_filename} generated.")
            else:
                print(f" Error generating log for λ = {lam}:")
                print(result.stderr)
            
            time.sleep(1) # Small delay between runs
            
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Restore original config file
        if original_config_content:
            with open(CONFIG_FILE, 'w') as f:
                f.write(original_config_content)
            print("\n---  Original config.yml restored ---")

def visualize_combined_load():
    """
    Reads the generated logs and creates a combined visualization with two
    subplots, each showing the load dynamics for all elasticity values.
    """
    print("\n---  Starting Combined Visualization ---")
    
    # Calculate and print load statistics for each lambda
    print("\n---  Load Statistics ---")
    for lam in ELASTICITY_VALUES:
        log_file = f"simulation_lambda_{lam}.log"
        try:
            df = pd.read_csv(log_file)
            load_avg = df['Load'].mean()
            load_std = df['Load'].std()
            fee_avg = df['BasePrice'].mean()
            fee_std = df['BasePrice'].std()
            print(f"λ = {lam:4.1f}: Average Load = {load_avg:.4f}, Load Deviation = {load_std:.4f}")
            print(f"         Average Fee = {fee_avg:.4f}, Fee Deviation = {fee_std:.4f}")
        except FileNotFoundError:
            print(f"λ = {lam:4.1f}: ⚠️ Log file not found: {log_file}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 9), sharey=True)
    
    for i, ax in enumerate(axes):
        start, end = INTERVALS[i]
        ax.set_title(f'Blocks {start}-{end}', fontsize=30, fontweight='bold')
        
        for lam in ELASTICITY_VALUES:
            log_file = f"simulation_lambda_{lam}.log"
            try:
                df = pd.read_csv(log_file)
                interval_df = df[(df['BlockNumber'] >= start) & (df['BlockNumber'] <= end)]
                
                ax.plot(
                    interval_df['BlockNumber'], 
                    interval_df['Load'],
                    color=COLORS[lam],
                    linestyle=LINE_STYLES[lam],
                    linewidth=2,
                    alpha=1.0,
                    label=f'λ = {lam}'
                )
            except FileNotFoundError:
                print(f" Warning: Log file not found: {log_file}")
        
        # Add target load line
        ax.axhline(y=0.5, color='black', linestyle='-.', linewidth=3, alpha=1.0, label='Target Load')
        
        # Styling
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Block Number', fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=26)
        ax.set_ylim(0, 1.1)
        
        # Set custom ticks
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        if i == 0:  # First subplot (0-50)
            ax.set_xticks([-2, 25, 50])
            ax.set_xticklabels(['0', '25', '50'])
        else:  # Second subplot (2500-2550)
            ax.set_xticks([2500, 2525, 2550])

    # Set shared Y-label
    axes[0].set_ylabel('Block Load', fontsize=28)
    
    # Create a single, unified legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), 
               ncol=4, fontsize=30, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95]) # Adjust layout to make space for legend
    
    # Save the plot
    output_filename = 'elasticity_load_comparison.pdf'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n--- Visualization saved as {output_filename} ---")
    plt.show()

if __name__ == '__main__':
    generate_simulation_logs()
    visualize_combined_load()
