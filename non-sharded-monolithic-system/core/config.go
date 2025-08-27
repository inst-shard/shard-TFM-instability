package core

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// YAMLConfig represents the structure of the config.yml file
type YAMLConfig struct {
	EIP1559 struct {
		TargetLoad       float64 `yaml:"target_load"`
		BaseFeeDelta     float64 `yaml:"base_fee_delta"`
		InitialBasePrice float64 `yaml:"initial_base_price"`
		GasLimit         int64   `yaml:"gas_limit"`
	} `yaml:"eip1559"`

	Demand struct {
		PriceElasticity float64 `yaml:"price_elasticity"`
		BaseDemand      float64 `yaml:"base_demand"`
	} `yaml:"demand"`

	Perturbation struct {
		Enabled    bool    `yaml:"enabled"`
		Amplitude  float64 `yaml:"amplitude"`
		Period     int     `yaml:"period"`
		StartBlock int     `yaml:"start_block"`
	} `yaml:"perturbation"`

	Simulation struct {
		Steps    int     `yaml:"steps"`
		TimeStep float64 `yaml:"time_step"`
	} `yaml:"simulation"`

	Shock struct {
		Enabled    bool `yaml:"enabled"`
		StartBlock int  `yaml:"start_block"`
		EndBlock   int  `yaml:"end_block"`
		Multiplier int  `yaml:"multiplier"`
	} `yaml:"shock"`

	Output struct {
		PrintInterval  int    `yaml:"print_interval"`
		DetailedOutput bool   `yaml:"detailed_output"`
		LogFile        string `yaml:"log_file"`
	} `yaml:"output"`
}

// Config contains all parameters for EIP-1559 simulation
type Config struct {
	// EIP-1559 core parameters
	TargetLoad       float64 // L_target: target load (e.g., 0.5)
	BaseFeeDelta     float64 // δ: price update rate (e.g., 0.125)
	InitialBasePrice float64 // P*: initial equilibrium price
	GasLimit         int64   // Gmax: block gas limit

	// Demand elasticity parameters
	PriceElasticity float64 // λ in T = C * P^(-λ)
	BaseDemand      float64 // C in T = C * P^(-λ)

	// Perturbation parameters to avoid false equilibrium
	PerturbationEnabled    bool    // whether to enable the perturbation
	PerturbationAmplitude  float64 // amplitude of the sine wave (e.g., 0.01 for 1%)
	PerturbationPeriod     int     // period of the sine wave in blocks
	PerturbationStartBlock int     // block number to start the perturbation

	// Simulation parameters
	SimulationSteps int     // number of simulation steps
	TimeStep        float64 // time step (e.g., 12 seconds = 1 block)

	// Shock parameters
	ShockEnabled    bool // whether to inject demand shock
	ShockStartBlock int  // block number to start shock
	ShockEndBlock   int  // block number to end shock (exclusive)
	ShockMultiplier int  // demand multiplier during shock

	// Output configuration
	PrintInterval  int    // print interval
	DetailedOutput bool   // whether to output detailed information
	LogFile        string // output log file name
}

// LoadConfig loads configuration from config.yml file
func LoadConfig(configFile string) (*Config, error) {
	data, err := os.ReadFile(configFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var yamlConfig YAMLConfig
	if err := yaml.Unmarshal(data, &yamlConfig); err != nil {
		return nil, fmt.Errorf("failed to parse YAML config: %w", err)
	}

	// Convert YAML config to internal Config struct
	config := &Config{
		// EIP-1559 parameters
		TargetLoad:       yamlConfig.EIP1559.TargetLoad,
		BaseFeeDelta:     yamlConfig.EIP1559.BaseFeeDelta,
		InitialBasePrice: yamlConfig.EIP1559.InitialBasePrice,
		GasLimit:         yamlConfig.EIP1559.GasLimit,

		// Demand parameters
		PriceElasticity: yamlConfig.Demand.PriceElasticity,
		BaseDemand:      yamlConfig.Demand.BaseDemand,

		// Perturbation parameters
		PerturbationEnabled:    yamlConfig.Perturbation.Enabled,
		PerturbationAmplitude:  yamlConfig.Perturbation.Amplitude,
		PerturbationPeriod:     yamlConfig.Perturbation.Period,
		PerturbationStartBlock: yamlConfig.Perturbation.StartBlock,

		// Simulation parameters
		SimulationSteps: yamlConfig.Simulation.Steps,
		TimeStep:        yamlConfig.Simulation.TimeStep,

		// Shock parameters
		ShockEnabled:    yamlConfig.Shock.Enabled,
		ShockStartBlock: yamlConfig.Shock.StartBlock,
		ShockEndBlock:   yamlConfig.Shock.EndBlock,
		ShockMultiplier: yamlConfig.Shock.Multiplier,

		// Output parameters
		PrintInterval:  yamlConfig.Output.PrintInterval,
		DetailedOutput: yamlConfig.Output.DetailedOutput,
		LogFile:        yamlConfig.Output.LogFile,
	}

	return config, nil
}

// DefaultConfig returns default configuration (kept for backward compatibility)
func DefaultConfig() *Config {
	config, err := LoadConfig("config.yml")
	if err != nil {
		// Fallback to hardcoded defaults if config file is not available
		return &Config{
			TargetLoad:       0.5,      // 50% target load
			BaseFeeDelta:     0.125,    // EIP-1559 standard adjustment rate
			InitialBasePrice: 1.0,      // 1 gwei
			GasLimit:         30000000, // 30M gas

			PriceElasticity: 0.8,      // λ = 0.8
			BaseDemand:      15000000, // base demand 15M gas

			PerturbationEnabled:    false,
			PerturbationAmplitude:  0.05, // 5% amplitude
			PerturbationPeriod:     100,  // 100 blocks period
			PerturbationStartBlock: 1,    // start perturbation at block 1

			SimulationSteps: 1000, // simulate 1000 steps
			TimeStep:        12.0, // 12 seconds per block

			ShockEnabled:    true,
			ShockStartBlock: 200,
			ShockEndBlock:   210,
			ShockMultiplier: 100,

			PrintInterval:  200,              // print every 200 steps
			DetailedOutput: false,            // no detailed output
			LogFile:        "simulation.log", // default log file
		}
	}
	return config
}
