package core

import (
	"math"
)

// DemandGenerator generates demand based on price, following the model in the user's document.
type DemandGenerator struct {
	config *Config
}

// NewDemandGenerator creates a new demand generator.
func NewDemandGenerator(config *Config) *DemandGenerator {
	return &DemandGenerator{
		config: config,
	}
}

// GenerateDemand generates demand based on current price using a power function.
// It can also apply a deterministic sine-wave perturbation to avoid false equilibrium.
// This directly implements the formula from the user's document: T = C * P^(-λ)
func (dg *DemandGenerator) GenerateDemand(currentPrice float64, blockNumber int) float64 {
	// We use a small epsilon to prevent the price from being zero.
	if currentPrice < 1e-9 {
		currentPrice = 1e-9
	}

	// Implements T = C * P^(-λ)
	// C is BaseDemand
	// λ is PriceElasticity (a positive value from config)
	demand := dg.config.BaseDemand * math.Pow(currentPrice, -dg.config.PriceElasticity)

	// Apply a deterministic perturbation if enabled and after the start block
	if dg.config.PerturbationEnabled && blockNumber >= dg.config.PerturbationStartBlock {
		// Calculate the sine wave perturbation
		// The wave completes one cycle every `PerturbationPeriod` blocks.
		// We adjust the angle to start the sine wave from 0 at the PerturbationStartBlock
		angle := 2 * math.Pi * float64(blockNumber-dg.config.PerturbationStartBlock) / float64(dg.config.PerturbationPeriod)
		perturbationFactor := dg.config.PerturbationAmplitude * math.Sin(angle)

		// Apply the perturbation to the base demand
		demand *= (1 + perturbationFactor)
	}

	// Ensure demand is not negative
	if demand < 0 {
		demand = 0
	}

	return demand
}
