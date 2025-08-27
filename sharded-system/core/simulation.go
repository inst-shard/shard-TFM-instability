package core

import (
	"fmt"
	"sort"
)

type Simulation struct {
	Config Config
	Shards map[int]*Shard
	step   int
}

func NewSimulation(config Config) *Simulation {
	simulation := &Simulation{
		Config: config,
		Shards: make(map[int]*Shard),
		step:   0,
	}

	// Initialize shards based on the configuration
	for _, spec := range config.Shards {
		simulation.Shards[spec.ID] = NewShard(spec)
	}
	// Set the simulation for each shard
	for _, shard := range simulation.Shards {
		shard.Simulation = simulation
	}
	return simulation
}

func (s *Simulation) Simulate() {
	s.GenerateInitialLoad()

	// Create sorted shard IDs for deterministic processing
	shardIDs := make([]int, 0, len(s.Shards))
	for shardID := range s.Shards {
		shardIDs = append(shardIDs, shardID)
	}
	sort.Ints(shardIDs) // Ensure deterministic order

	for step := 0; step < s.Config.Simulation.TotalSteps; step++ {
		s.step = step
		// Generate demand for each shard in deterministic order
		for _, shardID := range shardIDs {
			shard := s.Shards[shardID]
			demand := shard.GenerateDemand(step, &s.Config)
			shard.PendingLoad = append(shard.PendingLoad, demand...)
		}

		// Process loads for each shard and track individual shard loads
		shardLoads := make(map[int]float64)
		processedLoads := make(map[int][]*Load) // Store processed loads for analysis
		for _, shardID := range shardIDs {
			shardLoads[shardID] = 0.0
		}

		for _, shardID := range shardIDs {
			shard := s.Shards[shardID]
			result := shard.AbstractLoad(step)
			processedLoads[shard.Spec.ID] = result // Store the processed loads
			for _, load := range result {
				if load.Forward {
					targetShard := s.Shards[load.ShardTo]
					load.Forward = false // Mark as processed
					load.EffectiveStep = step + load.DelayedStep
					targetShard.PendingLoad = append(targetShard.PendingLoad, load)
				}
				// Add load to the originating shard's total
				shardLoads[shard.Spec.ID] += load.Amount
			}
		}

		// Update fees for each shard using their individual load ratios in deterministic order
		for _, shardID := range shardIDs {
			shard := s.Shards[shardID]
			loadRatio := shardLoads[shard.Spec.ID] / shard.Spec.GMax
			shard.UpdateFee(loadRatio)
		}

		// Print progress at key intervals
		if step%50 == 0 || step == s.Config.Simulation.TotalSteps-1 {
			fmt.Printf("Step %d: ", step)
			for _, shardID := range shardIDs {
				shard := s.Shards[shardID]
				fmt.Printf("Shard%d(%.3f) ", shardID, shardLoads[shardID]/shard.Spec.GMax)
			}
			fmt.Println()
		}

		// Print detailed load composition for steps 195-250
		if step >= 195 && step <= 250 {
			s.printLoadComposition(step, processedLoads)
		}
	}

}

func (s *Simulation) printLoadComposition(step int, processedLoads map[int][]*Load) {
	fmt.Printf("\n--- Step %d Load Composition ---\n", step)

	for shardID := 0; shardID < len(s.Shards); shardID++ {
		loads := processedLoads[shardID]

		// Analyze the processed loads
		totalLoad := 0.0
		localLoad := 0.0
		outboundLoad := 0.0
		inboundLoad := 0.0

		for _, load := range loads {
			if load == nil {
				continue
			}

			totalLoad += load.Amount

			if load.ShardFrom == load.ShardTo {
				localLoad += load.Amount
			} else if load.ShardFrom == shardID {
				outboundLoad += load.Amount
			} else {
				inboundLoad += load.Amount
			}
		}

		fmt.Printf("Shard %d: Total=%.0f, Local=%.0f, Outbound=%.0f, Inbound=%.0f\n",
			shardID, totalLoad, localLoad, outboundLoad, inboundLoad)
	}
}

func (s *Simulation) GenerateInitialLoad() {
	// Create sorted shard IDs for deterministic processing
	shardIDs := make([]int, 0, len(s.Shards))
	for shardID := range s.Shards {
		shardIDs = append(shardIDs, shardID)
	}
	sort.Ints(shardIDs) // Ensure deterministic order

	for _, shardID := range shardIDs {
		shard := s.Shards[shardID]

		for effectiveStep := 0; effectiveStep < s.Config.Delay.MaxDelay; effectiveStep++ {

			for sourceShardID := 0; sourceShardID < len(s.Config.Demand.BaseDemandMatrix); sourceShardID++ {
				if sourceShardID == shard.Spec.ID {
					continue // Skip local loads
				}

				baseDemand := s.Config.Demand.BaseDemandMatrix[sourceShardID][shard.Spec.ID]

				for delayIndex := effectiveStep; delayIndex < len(s.Config.Delay.Weights); delayIndex++ {
					weight := s.Config.Delay.Weights[delayIndex]
					if weight > 0 {
						initialLoad := &Load{
							ShardFrom:     sourceShardID,
							ShardTo:       shard.Spec.ID,
							Amount:        baseDemand * weight,
							Step:          effectiveStep - (delayIndex + 1), // Negative number indicates historical load
							DelayedStep:   delayIndex + 1,
							EffectiveStep: effectiveStep,
							Forward:       false,
						}
						shard.PendingLoad = append(shard.PendingLoad, initialLoad)
					}
				}
			}
		}
	}
}
