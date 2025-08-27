package core

import "sort"

type Shard struct {
	Spec        ShardSpec
	Simulation  *Simulation
	Load        []float64
	Fee         []float64
	PendingLoad []*Load
}

type Load struct {
	ShardFrom     int
	ShardTo       int
	Amount        float64
	Step          int
	DelayedStep   int
	EffectiveStep int
	Forward       bool
}

func NewShard(spec ShardSpec) *Shard {
	shard := &Shard{
		Spec:        spec,
		Load:        make([]float64, 0, 100000),
		Fee:         make([]float64, 0, 100000),
		PendingLoad: make([]*Load, 0, 100000),
	}

	// Initialize with base fee
	shard.Fee = append(shard.Fee, 1.0) // Base fee of 1.0

	return shard
}

func (s *Shard) UpdateFee(load float64) {
	if len(s.Fee) == 0 {
		s.Fee = append(s.Fee, 1.0) // Initialize with base fee if empty
	}
	fee := s.Fee[len(s.Fee)-1] * (1 + 0.25*(load-0.5))
	s.Fee = append(s.Fee, fee)
}

func (s *Shard) AbstractLoad(step int) []*Load {
	// Collect eligible loads
	var candidateLoads []*Load
	var candidateIndices []int

	for i, load := range s.PendingLoad {
		if load != nil && load.EffectiveStep <= step {
			// Create a copy for modification
			loadCopy := *load
			candidateLoads = append(candidateLoads, &loadCopy)
			candidateIndices = append(candidateIndices, i)
		}
	}

	// Sort by EffectiveStep in ascending order, keeping track of indices
	indices := make([]int, len(candidateLoads))
	for i := range indices {
		indices[i] = i
	}

	sort.Slice(indices, func(i, j int) bool {
		return candidateLoads[indices[i]].EffectiveStep < candidateLoads[indices[j]].EffectiveStep
	})

	// Create sorted loads and corresponding indices
	sortedLoads := make([]*Load, len(candidateLoads))
	sortedIndices := make([]int, len(candidateLoads))
	for i, idx := range indices {
		sortedLoads[i] = candidateLoads[idx]
		sortedIndices[i] = candidateIndices[idx]
	}

	// Calculate total amount
	totalAmount := 0.0
	for _, load := range sortedLoads {
		totalAmount += load.Amount
	}

	gmax := s.Spec.GMax

	// Case 1: If total amount is less than or equal to gmax, return all loads
	if totalAmount <= gmax {
		// Remove from PendingLoad (from back to front to maintain indices)
		sort.Sort(sort.Reverse(sort.IntSlice(sortedIndices)))
		for _, idx := range sortedIndices {
			s.PendingLoad = append(s.PendingLoad[:idx], s.PendingLoad[idx+1:]...)
		}
		return sortedLoads
	}

	// Case 2: Total amount is greater than gmax, need proportional allocation
	var result []*Load
	var toRemove []int
	remainingCapacity := gmax

	// Group by EffectiveStep
	stepGroups := make(map[int][]*Load)
	stepIndices := make(map[int][]int)
	for i, load := range sortedLoads {
		stepGroups[load.EffectiveStep] = append(stepGroups[load.EffectiveStep], load)
		stepIndices[load.EffectiveStep] = append(stepIndices[load.EffectiveStep], sortedIndices[i])
	}

	// Get sorted EffectiveStep list
	var sortedSteps []int
	for step := range stepGroups {
		sortedSteps = append(sortedSteps, step)
	}
	sort.Ints(sortedSteps)

	// Process each EffectiveStep in ascending order
	for _, effectiveStep := range sortedSteps {
		loads := stepGroups[effectiveStep]
		indices := stepIndices[effectiveStep]

		// Calculate total amount for current step
		stepTotalAmount := 0.0
		for _, load := range loads {
			stepTotalAmount += load.Amount
		}

		// If all loads in current step can be included
		if stepTotalAmount <= remainingCapacity {
			// Include all loads
			for _, load := range loads {
				result = append(result, load)
			}
			// Mark indices for removal
			toRemove = append(toRemove, indices...)
			remainingCapacity -= stepTotalAmount
		} else {
			// Include proportionally
			ratio := remainingCapacity / stepTotalAmount

			for i, load := range loads {
				originalIdx := indices[i]
				originalLoad := s.PendingLoad[originalIdx]

				// Create new partial load
				partialLoad := &Load{
					ShardFrom:     load.ShardFrom,
					ShardTo:       load.ShardTo,
					Amount:        load.Amount * ratio,
					Step:          load.Step,
					DelayedStep:   load.DelayedStep,
					EffectiveStep: load.EffectiveStep,
					Forward:       load.Forward,
				}
				result = append(result, partialLoad)

				// Update original load amount
				originalLoad.Amount -= partialLoad.Amount
				if originalLoad.Amount <= 0 {
					toRemove = append(toRemove, originalIdx)
				}
			}

			// Capacity exhausted, break the loop
			break
		}

		// If capacity is exhausted, break the loop
		if remainingCapacity <= 0 {
			break
		}
	}

	// Remove processed loads (from back to front to maintain indices)
	sort.Sort(sort.Reverse(sort.IntSlice(toRemove)))
	for _, idx := range toRemove {
		s.PendingLoad = append(s.PendingLoad[:idx], s.PendingLoad[idx+1:]...)
	}

	return result
}
