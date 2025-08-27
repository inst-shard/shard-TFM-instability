package main

import (
	"fmt"
	"gas-for-multiple-shards/core"
	"log"
	"sort"
)

func main() {
	// Load configuration
	config, err := core.LoadConfig("config.yml")
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	fmt.Printf("Enhanced Gas Fee Simulation - %d shards, %d steps\n",
		config.Network.NumShards, config.Simulation.TotalSteps)

	// Create and run simulation
	simulation := core.NewSimulation(*config)

	// Run simulation with enhanced logging
	logger := core.NewSimulationLogger(simulation)
	logger.RunAndLog()

	fmt.Printf("Enhanced analysis log generated: shard.log\n")
	fmt.Println("Run 'python3 visualize.py' to generate plots")

	// Final results
	fmt.Println("\nFinal Results:")
	// Create sorted shard IDs for deterministic output
	shardIDs := make([]int, 0, len(simulation.Shards))
	for shardID := range simulation.Shards {
		shardIDs = append(shardIDs, shardID)
	}
	sort.Ints(shardIDs)

	for _, shardID := range shardIDs {
		shard := simulation.Shards[shardID]
		finalFee := shard.Fee[len(shard.Fee)-1]
		fmt.Printf("Shard %d: Fee = %.6f\n", shardID, finalFee)
	}
}
