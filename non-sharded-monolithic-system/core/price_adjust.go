package core

// PriceAdjuster implements EIP-1559 price adjustment mechanism
type PriceAdjuster struct {
	config *Config
}

// NewPriceAdjuster creates a new price adjuster
func NewPriceAdjuster(config *Config) *PriceAdjuster {
	return &PriceAdjuster{
		config: config,
	}
}

// AdjustPrice implements EIP-1559 price adjustment: load(t) -> P(t+1)
// Formula: P(t+1) = P(t) * (1 + δ * 2 * (L(t) - L_target))
// The factor 2 ensures that when load goes from 0 to 1, price changes by ±12.5%
func (pa *PriceAdjuster) AdjustPrice(currentPrice, currentLoad float64) float64 {
	// Calculate load deviation from target
	loadDeviation := currentLoad - pa.config.TargetLoad

	// Apply EIP-1559 formula with factor 2 to match the standard ±12.5% range
	// When load = 0, loadDeviation = -0.5, adjustment = 1 + 0.125 * 2 * (-0.5) = 0.875 (-12.5%)
	// When load = 1, loadDeviation = +0.5, adjustment = 1 + 0.125 * 2 * (+0.5) = 1.125 (+12.5%)
	adjustmentFactor := 1.0 + pa.config.BaseFeeDelta*2*loadDeviation
	newPrice := currentPrice * adjustmentFactor

	return newPrice
}
