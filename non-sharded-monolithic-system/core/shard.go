package core

import (
	"fmt"
	"math"
)

// Shard represents the state of a single shard.
type Shard struct {
	// Current state
	CurrentBasePrice float64 // P(t): Current base price for the next block
	CurrentLoad      float64 // L(t): Load of the last processed block (0-1)
	PendingGas       float64 // Gas from transactions waiting in the queue

	// Configuration and components
	config          *Config
	priceAdjuster   *PriceAdjuster
	demandGenerator *DemandGenerator

	// Simulation stats
	BlockNumber int64
}

// NewShard creates a new shard.
func NewShard(config *Config) *Shard {
	shard := &Shard{
		config:           config,
		CurrentBasePrice: config.InitialBasePrice,
		CurrentLoad:      config.TargetLoad, // Start at target load
		PendingGas:       0,
		BlockNumber:      0,
	}

	// Initialize components
	shard.priceAdjuster = NewPriceAdjuster(config)
	shard.demandGenerator = NewDemandGenerator(config)

	return shard
}

// ProcessBlock processes one block, implementing the full simulation loop.
func (s *Shard) ProcessBlock() {
	s.BlockNumber++

	// 1. A new block arrives. Users see the current price P(t) and generate demand.
	// P(t) -> demand(t)
	newDemand := s.demandGenerator.GenerateDemand(s.CurrentBasePrice, int(s.BlockNumber))

	// 2. Total demand for this block is the new demand plus any pending gas.
	totalDemand := s.PendingGas + newDemand

	// 3. Determine how much gas is actually included in this block.
	gasUsedInBlock := math.Min(totalDemand, float64(s.config.GasLimit))

	// 4. Update the pending gas queue for the next block.
	s.PendingGas = totalDemand - gasUsedInBlock

	// 5. Calculate the load of the block that was just processed.
	// demand(t) -> load(t)
	s.CurrentLoad = gasUsedInBlock / float64(s.config.GasLimit)

	// 6. Adjust the price for the *next* block based on the current block's load.
	// load(t) -> P(t+1)
	newPrice := s.priceAdjuster.AdjustPrice(s.CurrentBasePrice, s.CurrentLoad)
	s.CurrentBasePrice = newPrice
}

// GetStatus returns a string with the current status of the shard.
func (s *Shard) GetStatus() string {
	return fmt.Sprintf(
		"Block %-4d | Price: %-7.2f | Load: %.2f%% | Gas Used: %-9.0f | Pending Gas: %-9.0f",
		s.BlockNumber,
		s.CurrentBasePrice,
		s.CurrentLoad*100,
		s.CurrentLoad*float64(s.config.GasLimit),
		s.PendingGas,
	)
}
