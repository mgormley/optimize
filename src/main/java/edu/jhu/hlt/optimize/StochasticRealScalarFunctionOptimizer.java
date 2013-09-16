package edu.jhu.hlt.optimize;


/**
 * All stochastic scalar function optimizers extend this class.
 * 
 * @author Nicholas Andrews
 */
public abstract class StochasticRealScalarFunctionOptimizer implements StochasticRealScalarFunction {
	StochasticRealScalarFunction f;
	public StochasticRealScalarFunctionOptimizer(StochasticRealScalarFunction f) {
		this.f = f;
	}
	public abstract void optimize();
}
