# Sharded EIP-1559 Instability Research Artifact

## Overview

This repository contains the implementation and simulation code for analyzing EIP-1559 transaction fee mechanisms in both monolithic and sharded blockchain systems. The project includes two main components: a baseline monolithic system and a comprehensive sharded system with various experimental configurations.

## Repository Structure

```
├── non-sharded-monolithic-system/     # Section 4: Baseline monolithic EIP-1559 analysis
│   ├── main.go                        # Single-shard simulation engine
│   ├── config.yml                     # Configuration for monolithic experiments
│   ├── core/                          # Core EIP-1559 implementation
│   └── simulation/                    # Figure 1 generation
│
├── sharded-system/                    # Sections 5-7: Sharded system analysis
│   ├── main.go                        # Multi-shard simulation engine
│   ├── config.yml                     # Default sharded system configuration
│   ├── core/                          # Sharded EIP-1559 implementation
│   ├── simulations/
│   │   ├── fast-simulations/          # Quick validation experiments
│   │   │   ├── Figure3/               # Quick Over-correction analysis
│   │   │   ├── Figure4/               # Defense amplifier analysis
│   │   │   ├── Figure6/               # Attack amplifier analysis when varying delayed attack
│   │   │   ├── Figure7/               # Defense amplifier analysis when varying inst attack
│   │   │   └── Figure11/              # Defense amplifier analysis when varying delayed attack
│   │   │
│   │   └── time-consuming-simulations/ # Comprehensive boundary mapping
│   │       ├── Figure5/               # Over-correction stability boundaries
│   │       ├── Figure8/               # Phase map prediction
│   │       ├── Figure9-ab/            # Extended boundary analysis when d=3
│   │       ├── Figure9-cd/            # Extended boundary analysis when d=7
│   │       ├── Figure10/              # Scalability analysis (varying shard count)
│   │       ├── Figure12/              # Delayed attack boundaries
│   │       ├── Figure13/              # Inst attack boundaries
│   │       └── Figure14/              # Inst attack boundary when d=1
```

## System Components

### Non-Sharded Monolithic System
**Location**: `non-sharded-monolithic-system/`

This implements a baseline single-shard EIP-1559 system for comparison purposes.

**Key Files**:
- `main.go`: Single-shard simulation engine
- `core`: Core EIP-1559 pricing mechanism implementation
- `core/demand_generator.go`: User demand modeling with price elasticity
- `simulation/Figure1/`: Baseline stability demonstration

### Sharded System
**Location**: `sharded-system/`

This implements a multi-shard system with cross-shard transaction support and various experimental configurations.

**Key Files**:
- `main.go`: Multi-shard simulation engine
- `core/simulation.go`: Sharded system coordination
- `core/shard.go`: Individual shard implementation with cross-shard coupling
- `core/demand_generator.go`: Cross-shard demand modeling

#### Fast Simulations (`fast-simulations/`)
Quick experiments for demonstrating specific phenomena:
- **Figure 1**: References monolithic baseline
- **Figure 3**: Quick demonstration
- **Figure 4**: Defense amplifier analysis
- **Figure 6**: Attack amplifier analysis
- **Figure 7**: Defense amplifier analysis when varying inst attack
- **Figure 11**: Defense amplifier analysis when varying delayed attack

#### Time-Consuming Simulations (`time-consuming-simulations/`)
Comprehensive parameter space exploration:
- **Figure 5**: Over-correction stability boundaries under different delay distributions
- **Figure 8**: Phase map prediction
- **Figure 9-ab/9-cd**: Extended boundary analysis with latency varying parameters
- **Figure 10**: Scalability analysis with varying shard counts (3, 5, 7, 10 shards)
- **Figure 12**: Delayed attack boundary mapping
- **Figure 13**: Inst attack boundaries
- **Figure 14**: Inst attack boundary when d=1

## System Requirements

### Software Dependencies
- **Go**: Version 1.24 or higher
- **Python**: Version 3.13.4 or higher
- **Python Packages**: 
  ```bash
  pip install numpy matplotlib pandas seaborn
  ```

### Hardware Requirements
- No specific hardware requirements

## Quick Start Guide

### 1. Monolithic System Testing

```bash
# Navigate to monolithic system
cd non-sharded-monolithic-system

# Run baseline simulation
go run main.go

# Generate Figure 1
cd simulation/Figure1
python generate_and_plot.py
```

**Expected Output**: 
- `simulation.log`: Time series data
- Figure 1: Load and price convergence plots

### 2. Sharded System Testing

#### Fast Simulations (Quick Validation)

```bash
# Navigate to sharded system
cd sharded-system

# Test basic sharded simulation
go run main.go

# Run specific fast simulations
cd simulations/fast-simulations/Figure3
python diagram-test-over-correction.py

cd ../Figure4
python kappa-defense-delay-defense.py

cd ../Figure6
python kappa-attack-delay-attack.py
```

**Expected Output**:
- Demonstration of system behavior under sharded conditions
- Visualization of amplification factors

#### Time-Consuming Simulations (Comprehensive Analysis)

```bash
# Navigate to time-consuming simulations
cd sharded-system/simulations/time-consuming-simulations

# Run over-correction boundary mapping (Figure 5)
cd Figure5
python over-correction-convergence-boundary-experiment.py
python over-correction-bubble-diagram.py

# Run scalability analysis (Figure 10)
cd ../Figure10
python experiment-varying-shard-number.py
python over-correction-bubble-diagram.py

# Run comprehensive stability mapping (Figure 12-14)
cd ../Figure12
python boundary-delay-attack.py
python attack-defense-bubble-diagram.py
```

**Expected Output**:
- Complete stability boundary maps
- Load inflation measurements
- Convergence ratio analysis across parameter spaces

## Configuration Parameters

### Monolithic System Configuration (`non-sharded-monolithic-system/config.yml`)

```yaml
eip1559:
  target_load: 0.5          # L_target: 50% target load
  base_fee_delta: 0.125     # δ: EIP-1559 update rate
  initial_base_price: 1.0   # P*: equilibrium price
  gas_limit: 30000000       # Block gas limit

demand:
  price_elasticity: 4.4     # λ: demand price elasticity
  base_demand: 15000000     # Base demand level

shock:
  enabled: true             # Enable demand shock testing
  start_block: 1            # Shock start block
  end_block: 2              # Shock end block
  multiplier: 15            # 15x demand spike
```

### Sharded System Configuration (`sharded-system/config.yml`)

```yaml
simulation:
  total_steps: 5000         # Simulation length
  l_target: 0.5             # Target load (50%)
  delta: 0.125              # EIP-1559 update rate

network:
  num_shards: 3             # Number of shards

demand:
  base_demand_matrix:       # Cross-shard transaction matrix
    - [300000, 350000, 350000]  # Shard 0 demand distribution
    - [0, 300000, 175000]       # Shard 1 demand distribution
    - [0, 175000, 300000]       # Shard 2 demand distribution

# this demand distribution is mapped in each experiment
  epsilon_matrix:           # Cross-shard price elasticity (ε_ij)
    - [0.0, 1.5, 1.5]
    - [1.5, 0.0, 1.5]
    - [1.5, 1.5, 0.0]

  lambda_matrix:            # Source price elasticity (λ_ij)
    - [1.5, 1.5, 1.5]
    - [1.5, 1.5, 1.5]
    - [1.5, 1.5, 1.5]

delay:
  max_delay: 5              # Maximum cross-shard delay
  weights: [0, 0, 0, 0, 1]  # Delay distribution (spike at 5 blocks)
```

## Key Experimental Parameters

### Critical Parameters
1. **Price Elasticity (λ, ε)**: Controls demand sensitivity to price changes
2. **Cross-shard Traffic Ratios (α)**: Proportion of cross-shard transactions
3. **Delay Distribution (w_d)**: Network latency structure
   - **Spike distribution**: Concentrated delay at specific block
   - **Uniform distribution**: Evenly distributed delay
   - **Bimodal distribution**: Two-peak delay distribution

## Understanding the Results

### Output Metrics

1. **Convergence Ratio (γ)**: Percentage of parameter space where system remains stable
2. **Load Inflation**: Average load deviation from 50% target
   - Formula: (AvgLoad/L_target) - 1
3. **Load Standard Deviation**: Measure of system volatility

### Key Measurements
- **Amplification Factors (κ)**: Delay-induced amplification effects

## Reproducing Paper Figures

### Figure 1: Monolithic Baseline
```bash
cd non-sharded-monolithic-system/simulation/Figure1
python generate_and_plot.py
```

### Figures 3-4, 6-7, 11: Fast Analysis
```bash
cd sharded-system/simulations/fast-simulations/Figure[X]
python [corresponding-script].py
```

### Figures 5, 8-9, 12-14: Comprehensive Analysis
```bash
cd sharded-system/simulations/time-consuming-simulations/Figure[X]
python [experiment-script].py
python [visualization-script].py
```

## Troubleshooting

### Common Issues

1. **Go Module Issues**
   ```bash
   go mod tidy
   go mod download
   ```

2. **Python Dependencies**
   ```bash
   pip install --upgrade numpy matplotlib pandas seaborn
   ```


3. **Configuration Errors**
   - Verify YAML syntax in config files
   - Check matrix dimensions match number of shards
   - Ensure delay weights sum to 1.0


This implementation provides a complete simulation framework for analyzing EIP-1559 behavior in both monolithic and sharded environments. All experiments can be reproduced using the provided configurations and scripts.
