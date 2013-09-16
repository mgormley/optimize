package edu.jhu.hlt.optimize;


/**
 * All stochastic scalar function optimizers extend this class.
 * 
 * @author Nicholas Andrews
 */
public abstract class StochasticRealScalarFunctionOptimizer implements StochasticRealScalarFunction {
	RealScalarFunction f;
	public StochasticRealScalarFunctionOptimizer(RealScalarFunction f) {
		this.f = f;
	}
	public RealScalarFunction getFunction() { return f; }
	public abstract void optimize();
}
