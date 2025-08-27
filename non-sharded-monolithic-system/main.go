package main

import (
	"encoding/csv"
	"fmt"
	"gas-for-one-shard/core"
	"log"
	"os"
	"strconv"
)

func main() {
	// --- Load Configuration ---
	config, err := core.LoadConfig("config.yml")
	if err != nil {
		log.Printf("Warning: Failed to load config.yml, using defaults: %v", err)
		config = core.DefaultConfig()
	}

	shard := core.NewShard(config)

	// --- Logging Setup ---
	file, err := os.Create(config.LogFile)
	if err != nil {
		log.Fatalf("Failed to create log file: %s", err)
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write CSV Header
	header := []string{"BlockNumber", "BasePrice", "Load", "GasUsed", "PendingGas"}
	if err := writer.Write(header); err != nil {
		log.Fatalf("Failed to write header to log file: %s", err)
	}

	fmt.Println("--- EIP-1559 Single Shard Simulation Start ---")
	fmt.Printf("Running simulation and generating log file: %s\n", config.LogFile)

	// --- Simulation Loop ---
	for i := 1; i <= config.SimulationSteps; i++ {
		// Apply demand shock if configured. The check must use the canonical block number from the shard (`shard.BlockNumber`),
		// not the loop counter `i`, because `i` is just a step counter and doesn't represent the block time.
		if config.ShockEnabled && shard.BlockNumber >= int64(config.ShockStartBlock) && shard.BlockNumber < int64(config.ShockEndBlock) {
			if shard.BlockNumber == int64(config.ShockStartBlock) { // Print message only at the start of the shock period
				fmt.Printf("\n!!! Injecting %dx demand shock from Block %d to %d !!!\n",
					config.ShockMultiplier, config.ShockStartBlock, config.ShockEndBlock-1)
			}
			originalDemand := config.BaseDemand
			config.BaseDemand *= float64(config.ShockMultiplier)
			shard.ProcessBlock()
			config.BaseDemand = originalDemand // Reset to normal
		} else {
			shard.ProcessBlock()
		}

		// --- Log data for every block ---
		gasUsed := shard.CurrentLoad * float64(config.GasLimit)
		logData := []string{
			strconv.FormatInt(shard.BlockNumber, 10),
			strconv.FormatFloat(shard.CurrentBasePrice, 'f', 4, 64),
			strconv.FormatFloat(shard.CurrentLoad, 'f', 4, 64),
			strconv.FormatFloat(gasUsed, 'f', 0, 64),
			strconv.FormatFloat(shard.PendingGas, 'f', 0, 64),
		}
		if err := writer.Write(logData); err != nil {
			log.Fatalf("Failed to write to log file: %s", err)
		}

		// Print status to console periodically
		if shard.BlockNumber%int64(config.PrintInterval) == 0 {
			fmt.Println(shard.GetStatus())
		}
	}

	fmt.Println("\n--- Simulation End ---")
	fmt.Printf("Log file '%s' has been generated.\n", config.LogFile)
	fmt.Println("Next, run 'python visualize.py' to see the results.")
}
