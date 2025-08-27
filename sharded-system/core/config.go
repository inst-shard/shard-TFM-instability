package core

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Config is the root struct for the entire configuration file.
type Config struct {
	Simulation SimulationConfig `yaml:"simulation"`
	Network    NetworkConfig    `yaml:"network"`
	Shards     []ShardSpec      `yaml:"shards"`
	Demand     DemandConfig     `yaml:"demand"`
	Delay      DelayConfig      `yaml:"delay"`
}

// SimulationConfig contains the global parameters for the simulation.
type SimulationConfig struct {
	TotalSteps   int                `yaml:"total_steps"`
	LTarget      float64            `yaml:"l_target"`
	Delta        float64            `yaml:"delta"`
	Perturbation PerturbationConfig `yaml:"perturbation"`
	DemandShock  DemandShockConfig  `yaml:"demand_shock"`
}

// PerturbationConfig contains parameters for the sinusoidal demand perturbation.
type PerturbationConfig struct {
	Enabled   bool    `yaml:"enabled"`
	Amplitude float64 `yaml:"amplitude"`
	Period    int     `yaml:"period"`
}

// DemandShockConfig contains parameters for a one-time demand shock.
type DemandShockConfig struct {
	Enabled      bool    `yaml:"enabled"`
	StartStep    int     `yaml:"start_step"`
	EndStep      int     `yaml:"end_step"`
	Multiplier   float64 `yaml:"multiplier"`
	TargetShards []int   `yaml:"target_shards"`
}

// NetworkConfig contains network definitions.
type NetworkConfig struct {
	NumShards int `yaml:"num_shards"`
}

// ShardSpec contains the static parameters for each shard.
type ShardSpec struct {
	ID   int     `yaml:"id"`
	GMax float64 `yaml:"g_max"`
}

// DemandConfig contains parameters related to the demand function.
type DemandConfig struct {
	BaseDemandMatrix [][]float64 `yaml:"base_demand_matrix"`
	LambdaMatrix     [][]float64 `yaml:"lambda_matrix"`
	EpsilonMatrix    [][]float64 `yaml:"epsilon_matrix"`
}

// DelayConfig contains parameters for cross-shard transaction delays.
type DelayConfig struct {
	MaxDelay int       `yaml:"max_delay"`
	Weights  []float64 `yaml:"weights"`
}

// LoadConfig loads and parses the YAML configuration file from the specified path.
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config Config
	err = yaml.Unmarshal(data, &config)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal config YAML: %w", err)
	}

	if err := validateConfig(&config); err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}

	return &config, nil
}

// validateConfig checks the logical consistency of the configuration.
func validateConfig(c *Config) error {
	numShards := c.Network.NumShards
	if numShards <= 0 {
		return fmt.Errorf("num_shards must be positive")
	}

	if len(c.Shards) != numShards {
		return fmt.Errorf("number of items in shards list (%d) does not match num_shards (%d)", len(c.Shards), numShards)
	}

	if err := validateMatrix(c.Demand.BaseDemandMatrix, numShards, "base_demand_matrix"); err != nil {
		return err
	}
	if err := validateMatrix(c.Demand.LambdaMatrix, numShards, "lambda_matrix"); err != nil {
		return err
	}
	if err := validateMatrix(c.Demand.EpsilonMatrix, numShards, "epsilon_matrix"); err != nil {
		return err
	}

	if c.Delay.MaxDelay != len(c.Delay.Weights) {
		return fmt.Errorf("max_delay (%d) does not match the number of weights (%d)", c.Delay.MaxDelay, len(c.Delay.Weights))
	}

	// More validations can be added here, e.g., checking if the sum of weights is 1.

	return nil
}

// validateMatrix is a helper function to validate the dimensions of a matrix.
func validateMatrix(matrix [][]float64, expectedDim int, name string) error {
	if len(matrix) != expectedDim {
		return fmt.Errorf("%s has %d rows, expected %d", name, len(matrix), expectedDim)
	}
	for i, row := range matrix {
		if len(row) != expectedDim {
			return fmt.Errorf("row %d of %s has %d columns, expected %d", i, name, len(row), expectedDim)
		}
	}
	return nil
}
