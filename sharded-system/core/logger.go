package core

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"time"
)

type SimulationLogger struct {
	simulation *Simulation
}

type SimulationStep struct {
	Step      int
	ShardData map[int]ShardStepData
}

type ShardStepData struct {
	TotalLoad           float64
	LoadRatio           float64
	Fee                 float64
	ProcessedLoads      []*Load
	PendingLoadAnalysis PendingLoadAnalysis
}

type PendingLoadAnalysis struct {
	TotalPending    float64
	LocalPending    float64
	OutboundPending float64
	InboundPending  float64
	EffectiveSteps  map[int]float64
}

func NewSimulationLogger(simulation *Simulation) *SimulationLogger {
	return &SimulationLogger{
		simulation: simulation,
	}
}

func (sl *SimulationLogger) RunAndLog() string {
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("enhanced_simulation_analysis_%s.log", timestamp)

	file, err := os.Create(filename)
	if err != nil {
		panic(fmt.Sprintf("Failed to create log file: %v", err))
	}
	defer file.Close()

	// Write header
	sl.writeLogHeader(file)

	// Store simulation data
	var simulationData []SimulationStep

	// Run simulation with data collection
	sl.simulation.GenerateInitialLoad()

	// Create sorted shard IDs for deterministic processing
	shardIDs := make([]int, 0, len(sl.simulation.Shards))
	for shardID := range sl.simulation.Shards {
		shardIDs = append(shardIDs, shardID)
	}
	sort.Ints(shardIDs) // Ensure deterministic order

	for step := 0; step < sl.simulation.Config.Simulation.TotalSteps; step++ {
		stepData := SimulationStep{
			Step:      step,
			ShardData: make(map[int]ShardStepData),
		}

		// Generate demand for each shard in deterministic order
		for _, shardID := range shardIDs {
			shard := sl.simulation.Shards[shardID]
			demand := shard.GenerateDemand(step, &sl.simulation.Config)
			shard.PendingLoad = append(shard.PendingLoad, demand...)
		}

		// Collect pending load data before processing in deterministic order
		for _, shardID := range shardIDs {
			shard := sl.simulation.Shards[shardID]
			pendingAnalysis := sl.analyzePendingLoad(shard, step)
			stepData.ShardData[shardID] = ShardStepData{
				PendingLoadAnalysis: pendingAnalysis,
			}
		}

		// Process loads for each shard
		shardLoads := make(map[int]float64)
		processedLoads := make(map[int][]*Load)

		for _, shardID := range shardIDs {
			shardLoads[shardID] = 0.0
		}

		for _, shardID := range shardIDs {
			shard := sl.simulation.Shards[shardID]
			result := shard.AbstractLoad(step)
			processedLoads[shard.Spec.ID] = result

			for _, load := range result {
				if load.Forward {
					targetShard := sl.simulation.Shards[load.ShardTo]
					load.Forward = false
					load.EffectiveStep = step + load.DelayedStep
					targetShard.PendingLoad = append(targetShard.PendingLoad, load)
				}
				shardLoads[shard.Spec.ID] += load.Amount
			}
		}

		// Update fees in deterministic order
		for _, shardID := range shardIDs {
			shard := sl.simulation.Shards[shardID]
			loadRatio := shardLoads[shardID] / shard.Spec.GMax
			shard.UpdateFee(loadRatio)
		}

		// Complete step data in deterministic order
		for _, shardID := range shardIDs {
			shard := sl.simulation.Shards[shardID]
			data := stepData.ShardData[shardID]
			data.TotalLoad = shardLoads[shardID]
			data.LoadRatio = shardLoads[shardID] / shard.Spec.GMax
			data.Fee = shard.Fee[len(shard.Fee)-1]
			data.ProcessedLoads = processedLoads[shardID]
			stepData.ShardData[shardID] = data
		}

		simulationData = append(simulationData, stepData)

		// Progress indicator
		if step%50 == 0 {
			fmt.Printf("Processing step %d/%d\n", step, sl.simulation.Config.Simulation.TotalSteps)
		}
	}

	// Write detailed analysis to log
	sl.writeDetailedAnalysis(file, simulationData)

	// Write data section for Python visualization
	sl.writeDataSection(file, simulationData)

	return filename
}

func (sl *SimulationLogger) analyzePendingLoad(shard *Shard, currentStep int) PendingLoadAnalysis {
	analysis := PendingLoadAnalysis{
		EffectiveSteps: make(map[int]float64),
	}

	for _, load := range shard.PendingLoad {
		if load == nil {
			continue
		}

		// Only count loads with EffectiveStep <= currentStep
		if load.EffectiveStep <= currentStep {
			analysis.TotalPending += load.Amount
			analysis.EffectiveSteps[load.EffectiveStep] += load.Amount

			if load.ShardFrom == load.ShardTo {
				analysis.LocalPending += load.Amount
			} else if load.ShardFrom == shard.Spec.ID {
				analysis.OutboundPending += load.Amount
			} else {
				analysis.InboundPending += load.Amount
			}
		}
	}

	return analysis
}

func (sl *SimulationLogger) writeLogHeader(file *os.File) {
	config := sl.simulation.Config
	fmt.Fprintf(file, "╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n")
	fmt.Fprintf(file, "║                                                                              ENHANCED MULTI-SHARD GAS FEE SIMULATION ANALYSIS                                                                                        ║\n")
	fmt.Fprintf(file, "║                                                                                    Generated: %s                                                                                      ║\n", time.Now().Format("2006-01-02 15:04:05"))
	fmt.Fprintf(file, "╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n")
	fmt.Fprintf(file, "║ CONFIGURATION:                                                                                                                                                                                                              ║\n")
	fmt.Fprintf(file, "║ • Network: %d shards, each with capacity %s gas                                                                                                                                                                      ║\n",
		config.Network.NumShards, formatNumber(int(config.Shards[0].GMax)))
	fmt.Fprintf(file, "║ • Simulation Steps: %d                                                                                                                                                                                                     ║\n", config.Simulation.TotalSteps)
	fmt.Fprintf(file, "║ • Demand Shock: Steps %d-%d, %.1fx multiplier, targeting Shards %v                                                                                                                                                          ║\n",
		config.Simulation.DemandShock.StartStep, config.Simulation.DemandShock.EndStep, config.Simulation.DemandShock.Multiplier, config.Simulation.DemandShock.TargetShards)
	fmt.Fprintf(file, "║ • Enhanced Analysis: PendingLoad composition + EffectiveStep breakdown                                                                                                                                                     ║\n")
	fmt.Fprintf(file, "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n\n")
}

func (sl *SimulationLogger) writeDetailedAnalysis(file *os.File, data []SimulationStep) {
	// Write table header
	fmt.Fprintf(file, "┌──────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐\n")
	fmt.Fprintf(file, "│ STEP │                                                                    ENHANCED SHARD ANALYSIS                                                                                                                            │\n")
	fmt.Fprintf(file, "├──────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤\n")
	fmt.Fprintf(file, "│      │                           SHARD 0                           │                           SHARD 1                           │                           SHARD 2                           │\n")
	fmt.Fprintf(file, "│      │ Fee/Load/Ratio │  Pending Analysis  │  Effective Steps   │ Fee/Load/Ratio │  Pending Analysis  │  Effective Steps   │ Fee/Load/Ratio │  Pending Analysis  │  Effective Steps   │\n")
	fmt.Fprintf(file, "├──────┼────────────────┼────────────────────┼────────────────────┼────────────────┼────────────────────┼────────────────────┼────────────────┼────────────────────┼────────────────────┤\n")

	// Write data rows (sample every 5 steps to keep output manageable)
	for i, step := range data {
		if i%5 != 0 && i < len(data)-1 {
			continue
		}

		fmt.Fprintf(file, "│ %4d │", step.Step)

		for shardID := 0; shardID < 3; shardID++ {
			shardData := step.ShardData[shardID]

			// Fee/Load/Ratio
			fmt.Fprintf(file, " %5.3f/%5.0fk/%4.1f%% │",
				shardData.Fee,
				shardData.TotalLoad/1000,
				shardData.LoadRatio*100)

			// Pending Analysis
			fmt.Fprintf(file, " L:%3.0fk O:%3.0fk I:%3.0fk │",
				shardData.PendingLoadAnalysis.LocalPending/1000,
				shardData.PendingLoadAnalysis.OutboundPending/1000,
				shardData.PendingLoadAnalysis.InboundPending/1000)

			// Effective Steps (show top 3 steps with loads)
			fmt.Fprintf(file, " %s │", sl.formatEffectiveSteps(shardData.PendingLoadAnalysis.EffectiveSteps, step.Step))
		}

		fmt.Fprintf(file, "\n")

		// Add special markers for shock periods
		if step.Step >= sl.simulation.Config.Simulation.DemandShock.StartStep && step.Step <= sl.simulation.Config.Simulation.DemandShock.EndStep {
			fmt.Fprintf(file, "│      │              ⚡ DEMAND SHOCK PERIOD ⚡                                                                                                                                                          │\n")
		}
	}

	fmt.Fprintf(file, "└──────┴────────────────┴────────────────────┴────────────────────┴────────────────┴────────────────────┴────────────────────┴────────────────┴────────────────────┴────────────────────┘\n\n")

	// Write summary analysis
	sl.writeSummaryAnalysis(file, data)
}

func (sl *SimulationLogger) formatEffectiveSteps(steps map[int]float64, currentStep int) string {
	if len(steps) == 0 {
		return "                  "
	}

	// Find top 2 steps by load amount
	type stepLoad struct {
		step int
		load float64
	}

	var stepLoads []stepLoad
	for step, load := range steps {
		if load > 1000 { // Only show significant loads
			stepLoads = append(stepLoads, stepLoad{step, load})
		}
	}

	if len(stepLoads) == 0 {
		return "                  "
	}

	// Sort by load amount (descending)
	for i := 0; i < len(stepLoads)-1; i++ {
		for j := i + 1; j < len(stepLoads); j++ {
			if stepLoads[i].load < stepLoads[j].load {
				stepLoads[i], stepLoads[j] = stepLoads[j], stepLoads[i]
			}
		}
	}

	// Format top 2
	result := ""
	for i := 0; i < len(stepLoads) && i < 2; i++ {
		if i > 0 {
			result += " "
		}
		result += fmt.Sprintf("%d:%3.0fk", stepLoads[i].step, stepLoads[i].load/1000)
	}

	// Pad to 18 characters
	for len(result) < 18 {
		result += " "
	}

	return result
}

func (sl *SimulationLogger) writeSummaryAnalysis(file *os.File, data []SimulationStep) {
	fmt.Fprintf(file, "╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n")
	fmt.Fprintf(file, "║                                                                                     CONVERGENCE ANALYSIS                                                                                                                       ║\n")
	fmt.Fprintf(file, "╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n")

	// Analyze convergence for each shard
	for shardID := 0; shardID < 3; shardID++ {
		fmt.Fprintf(file, "║ SHARD %d CONVERGENCE:                                                                                                                                                                                                      ║\n", shardID)

		// Calculate final averages (last 50 steps)
		finalSteps := 50
		if len(data) < finalSteps {
			finalSteps = len(data)
		}

		avgFee := 0.0
		avgLoad := 0.0

		for i := len(data) - finalSteps; i < len(data); i++ {
			shardData := data[i].ShardData[shardID]
			avgFee += shardData.Fee
			avgLoad += shardData.LoadRatio
		}

		avgFee /= float64(finalSteps)
		avgLoad /= float64(finalSteps)

		fmt.Fprintf(file, "║   Final %d steps average: Fee = %.6f (target: 1.000000), Load Ratio = %.6f (target: 0.500000)                                                                                                               ║\n",
			finalSteps, avgFee, avgLoad)

		feeDeviation := ((avgFee - 1.0) / 1.0) * 100
		loadDeviation := ((avgLoad - 0.5) / 0.5) * 100

		fmt.Fprintf(file, "║   Deviation from targets: Fee %+.2f%%, Load %+.2f%%                                                                                                                                                                      ║\n",
			feeDeviation, loadDeviation)
	}

	fmt.Fprintf(file, "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n\n")
}

func (sl *SimulationLogger) writeDataSection(file *os.File, data []SimulationStep) {
	fmt.Fprintf(file, "# SIMULATION DATA FOR PYTHON VISUALIZATION\n")
	fmt.Fprintf(file, "# Format: Step,Shard0_Fee,Shard1_Fee,Shard2_Fee,Shard0_Load,Shard1_Load,Shard2_Load\n")
	fmt.Fprintf(file, "DATA_START\n")
	fmt.Fprintf(file, "Step,Shard0_Fee,Shard1_Fee,Shard2_Fee,Shard0_Load,Shard1_Load,Shard2_Load\n")

	for _, step := range data {
		fmt.Fprintf(file, "%d", step.Step)
		for i := 0; i < 3; i++ {
			shardData := step.ShardData[i]
			fmt.Fprintf(file, ",%.6f", shardData.Fee)
		}
		for i := 0; i < 3; i++ {
			shardData := step.ShardData[i]
			fmt.Fprintf(file, ",%.6f", shardData.LoadRatio)
		}
		fmt.Fprintf(file, "\n")
	}
	fmt.Fprintf(file, "DATA_END\n")
}

func formatNumber(n int) string {
	str := fmt.Sprintf("%d", n)
	if len(str) <= 3 {
		return str
	}

	var result []string
	for i, c := range str {
		if i > 0 && (len(str)-i)%3 == 0 {
			result = append(result, ",")
		}
		result = append(result, string(c))
	}

	return strings.Join(result, "")
}
