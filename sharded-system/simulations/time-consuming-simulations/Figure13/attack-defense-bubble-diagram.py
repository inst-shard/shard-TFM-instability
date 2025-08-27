#!/usr/bin/env python3
"""
Simplified Bubble Chart Visualization
Only the first subplot from bubble chart: size=std dev, color=inflation
Generates separate PDF files for each delay distribution.
With convergence boundary curve instead of individual markers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

plt.rcParams['pdf.compression'] = 9     
plt.rcParams['pdf.fonttype'] = 42        
plt.rcParams['ps.fonttype']  = 42

# Configuration
DISTRIBUTIONS = ['Spike', 'Uniform', 'Bimodal']

def load_convergence_data():
    """Load convergence data for all three delay distributions"""
    data = {}
    
    for dist in DISTRIBUTIONS:
        filename = f'attack_defense_results_{dist}.csv'
        try:
            df = pd.read_csv(filename)
            print(f"✅ Loaded {filename}: {len(df)} records")
            data[dist] = df
        except FileNotFoundError:
            print(f"❌ File not found: {filename}")
            return None
    
    return data

def create_single_bubble_chart(df, distribution_name):
    """Create single bubble chart: size=std dev, color=inflation"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10.3))

    # Create the bubble chart with user-friendly colormap
    scatter = ax.scatter(
        df['epsilon_0j'],
        df['alpha_outflow'],
        s=df['load_std'] * 1500,   
        c=df['load_inflation'],    
        cmap='viridis',
        alpha=0.6,
        edgecolors='none',        
        linewidths=0,
        rasterized=True           
    )
    ax.set_rasterization_zorder(0)
    scatter.set_zorder(-1)

    # Draw convergence boundary curve instead of individual markers
    converged_df = df[df['converged']]
    diverged_df = df[~df['converged']]
    
    # Create boundary by finding the transition points
    epsilon_values = sorted(df['epsilon_0j'].unique())
    boundary_points = []
    
    for eps in epsilon_values:
        eps_data = df[df['epsilon_0j'] == eps].sort_values('alpha_outflow')
        converged_alphas = eps_data[eps_data['converged']]['alpha_outflow'].values
        diverged_alphas = eps_data[~eps_data['converged']]['alpha_outflow'].values
        
        if len(converged_alphas) > 0 and len(diverged_alphas) > 0:
            max_converged_alpha = converged_alphas.max()
            min_diverged_alpha = diverged_alphas.min()
            boundary_alpha = (max_converged_alpha + min_diverged_alpha) / 2
            boundary_points.append((eps, boundary_alpha))
        elif len(converged_alphas) > 0:
            boundary_points.append((eps, converged_alphas.max()))
        elif len(diverged_alphas) > 0:
            boundary_points.append((eps, diverged_alphas.min()))
    
    # Draw the boundary curve with enhanced visibility
    if len(boundary_points) > 1:
        boundary_x, boundary_y = zip(*boundary_points)
        ax.plot(boundary_x, boundary_y,
                color='darkred', linewidth=4, linestyle='-',
                alpha=0.9, zorder=10, label='Boundary')

    # Formatting
    ax.set_xlabel('$ε_{0j}$', fontsize=34, fontweight='bold')
    ax.set_ylabel('$\sumα_{0j,out}$', fontsize=34, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set y-axis range and format as percentage
    ax.set_ylim(0.45, 0.95)
    y_ticks = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    ax.set_yticks(y_ticks)
    x_ticks = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0]
    ax.set_xticks(x_ticks)
    ax.set_yticklabels([f'{int(y*100)}%' for y in y_ticks])
    
    # Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=27)
    
    # Add colorbar for load inflation
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Load Inflation', rotation=270, labelpad=35, fontsize=32, fontweight='bold')
    cbar.ax.tick_params(labelsize=25)


    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar.update_ticks()
    infl_max = float(df['load_inflation'].max())
    cbar.ax.text(
        0.5, 1.02,
        f"max = {infl_max:.3f}",
        transform=cbar.ax.transAxes,
        ha='center', va='bottom',
        fontsize=27.5, fontweight='bold'
    )
    
    # Create combined legend
    legend_elements = []
    legend_elements.append(plt.Line2D([0], [0], color='darkred', linewidth=4, label='Boundary'))

    total_points = len(df)
    converged_count = len(converged_df)
    convergence_ratio = converged_count / total_points if total_points else 0.0
    import matplotlib.patches as mpatches
    legend_elements.append(mpatches.Patch(color='none', label=f'$\gamma$: {convergence_ratio:.1%}'))

    # Add minimum diverged combination to legend
    diverged_df = df[~df['converged']]
    if len(diverged_df) > 0:
        min_diverged = diverged_df.loc[diverged_df[['epsilon_0j', 'alpha_outflow']].sum(axis=1).idxmin()]
        legend_elements.append(mpatches.Patch(
            color='none',
            label=f'Min Div: ε={min_diverged["epsilon_0j"]:.2f}, α={min_diverged["alpha_outflow"]:.2f}'
        ))


    std_max = float(df['load_std'].max())
    legend_elements.append(plt.Line2D(
        [0], [0], marker='o', color='w',
        markerfacecolor='gray', markersize=16, alpha=0.6,
        label=f'Std Dev: {std_max:.2f} (max)'
    ))
    
    ax.legend(handles=legend_elements, fontsize=32, loc='lower center', bbox_to_anchor=(0.46, 1))
    
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Save the plot
    output_filename = f'attack-defense-bubble-chart-{distribution_name.lower()}.pdf'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"💾 Bubble chart saved as: {output_filename}")
    
    plt.show()

def analyze_bubble_metrics(df, distribution_name):
    """Analyze metrics for the bubble chart"""
    print(f"\n📊 {distribution_name} Distribution - Bubble Chart Metrics")
    print("="*60)
    
    converged_df = df[df['converged']]
    diverged_df = df[~df['converged']]
    
    # Find the minimum diverged combination
    if len(diverged_df) > 0:
        min_diverged = diverged_df.loc[diverged_df[['epsilon_0j', 'alpha_outflow']].sum(axis=1).idxmin()]
        print(f"\n🔴 Minimum Non-Convergent Combination:")
        print(f"  ε_0j: {min_diverged['epsilon_0j']:.3f}")
        print(f"  α_0j,out: {min_diverged['alpha_outflow']:.3f} ({min_diverged['alpha_outflow']*100:.1f}%)")
        print(f"  Load Std: {min_diverged['load_std']:.6f}")
        print(f"  Load Inflation: {min_diverged['load_inflation']:.6f}")
    else:
        print(f"\n✅ All combinations converged for {distribution_name}")
    
    print(f"\nLoad Standard Deviation:")
    print(f"  Range: {df['load_std'].min():.6f} - {df['load_std'].max():.6f}")
    print(f"  Overall: {df['load_std'].mean():.6f} ± {df['load_std'].std():.6f}")
    if len(converged_df) > 0:
        print(f"  Converged: {converged_df['load_std'].mean():.6f} ± {converged_df['load_std'].std():.6f}")
    if len(diverged_df) > 0:
        print(f"  Diverged: {diverged_df['load_std'].mean():.6f} ± {diverged_df['load_std'].std():.6f}")
    
    print(f"\nLoad Inflation:")
    print(f"  Range: {df['load_inflation'].min():.6f} - {df['load_inflation'].max():.6f}")
    print(f"  Overall: {df['load_inflation'].mean():.6f} ± {df['load_inflation'].std():.6f}")
    if len(converged_df) > 0:
        print(f"  Converged: {converged_df['load_inflation'].mean():.6f} ± {converged_df['load_inflation'].std():.6f}")
    if len(diverged_df) > 0:
        print(f"  Diverged: {diverged_df['load_inflation'].mean():.6f} ± {diverged_df['load_inflation'].std():.6f}")
    
    # Calculate correlation
    correlation = df['load_std'].corr(df['load_inflation'])
    print(f"\nCorrelation between Std Dev and Inflation: {correlation:.4f}")

def main():
    """Main function"""
    print("🔬 Simplified Bubble Chart Visualization")
    print("Bubble Size = Load Standard Deviation, Color = Load Inflation")
    print("="*70)
    
    # Load data
    data_dict = load_convergence_data()
    if data_dict is None:
        print("❌ Data loading failed")
        return
    
    # Process each distribution separately
    for dist in DISTRIBUTIONS:
        print(f"\n🎨 Processing {dist} distribution...")
        df = data_dict[dist]
        
        # Analyze metrics
        analyze_bubble_metrics(df, dist)
        
        # Create visualization
        print(f"  Creating bubble chart...")
        create_single_bubble_chart(df, dist)
    
    print("\n✅ Bubble chart visualization complete!")
    print("\n📝 Generated Files:")
    print("  - attack-defense-bubble-chart-spike.pdf")
    print("  - attack-defense-bubble-chart-uniform.pdf")
    print("  - attack-defense-bubble-chart-bimodal.pdf")
    print("\n📊 Chart Legend:")
    print("  - X-axis: ε_j0 (epsilon parameter)")
    print("  - Y-axis: α_j→i,in (alpha inflow parameter)")
    print("  - Bubble Size: Load Standard Deviation (larger = more variation)")
    print("  - Bubble Color: Load Inflation (viridis colormap)")
    print("  - Dark red line: Convergence Boundary")

if __name__ == '__main__':
    main()
