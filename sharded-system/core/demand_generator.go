package core

import (
	"math"
)

func (s *Shard) GenerateDemand(step int, config *Config) []*Load {
	isShockTarget := false
	for _, targetShardID := range config.Simulation.DemandShock.TargetShards {
		if targetShardID == s.Spec.ID {
			isShockTarget = true
			break

		}
	}
	isShockActive := config.Simulation.DemandShock.Enabled &&
		step >= config.Simulation.DemandShock.StartStep &&
		step < config.Simulation.DemandShock.EndStep // Use < for end_step to make it inclusive

	shockMultiplier := 1.0
	if isShockTarget && isShockActive {
		shockMultiplier = config.Simulation.DemandShock.Multiplier
	}

	localDemand := s.GenerateLocalDemand(step, config, shockMultiplier)
	forwardDemand := s.GenerateForwardDemandWithInboundShock(step, config, shockMultiplier, isShockActive)
	return append(localDemand, forwardDemand...)
}

func (s *Shard) GenerateLocalDemand(step int, config *Config, shockMultiplier float64) []*Load {
	basedemand := config.Demand.BaseDemandMatrix[s.Spec.ID][s.Spec.ID] * shockMultiplier
	price := s.Fee[len(s.Fee)-1] // Use the last fee as the current price

	amount := basedemand * math.Pow(price, -config.Demand.LambdaMatrix[s.Spec.ID][s.Spec.ID])
	return []*Load{
		{
			ShardFrom:     s.Spec.ID,
			ShardTo:       s.Spec.ID,
			Amount:        amount, // Example logic for local
			Step:          step,
			DelayedStep:   0,
			EffectiveStep: step,
			// Example logic for effective step

			Forward: false,
		},
	}
}

func (s *Shard) GenerateForwardDemandWithInboundShock(step int, config *Config, shockMultiplier float64, isShockActive bool) []*Load {
	loads := make([]*Load, 0)
	for _, shard := range config.Shards {
		if shard.ID == s.Spec.ID {
			continue // Skip self
		}

		// 基础需求乘数：如果当前分片是shock目标，使用shockMultiplier
		baseMultiplier := shockMultiplier

		// 新增：如果目标分片是shock目标，且当前分片不是shock目标，
		// 那么当前分片发往目标分片的交易也应该增加
		// if !isShockTargetShard(s.Spec.ID, config.Simulation.DemandShock.TargetShards) &&
		// 	isShockTargetShard(shard.ID, config.Simulation.DemandShock.TargetShards) &&
		// 	isShockActive {
		// 	baseMultiplier = config.Simulation.DemandShock.Multiplier
		// }

		basedemand := config.Demand.BaseDemandMatrix[s.Spec.ID][shard.ID] * baseMultiplier
		sourcePrice := s.Fee[len(s.Fee)-1]                                                         // Use the last fee as the current price
		targetPrice := s.Simulation.Shards[shard.ID].Fee[len(s.Simulation.Shards[shard.ID].Fee)-1] // Use the last fee of the target shard
		totalAmount := basedemand * math.Pow(sourcePrice, -config.Demand.LambdaMatrix[s.Spec.ID][shard.ID]) *
			math.Pow(targetPrice, -config.Demand.EpsilonMatrix[s.Spec.ID][shard.ID])

		if totalAmount > 0 {
			// Normal delay distribution generation
			for delayStep := 1; delayStep <= config.Delay.MaxDelay; delayStep++ {
				weight := config.Delay.Weights[delayStep-1] // Weight array is 0-indexed
				if weight > 0 {
					partialAmount := totalAmount * weight
					loads = append(loads, &Load{
						ShardFrom:     s.Spec.ID,
						ShardTo:       shard.ID,
						Amount:        partialAmount,
						Step:          step,
						DelayedStep:   delayStep,
						EffectiveStep: step,
						Forward:       true,
					})
				}
			}
		}
	}
	return loads
}

// 辅助函数：检查某个分片是否是shock目标
func isShockTargetShard(shardID int, targetShards []int) bool {
	for _, targetShardID := range targetShards {
		if targetShardID == shardID {
			return true
		}
	}
	return false
}
