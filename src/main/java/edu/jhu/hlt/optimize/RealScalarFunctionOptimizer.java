package edu.jhu.hlt.optimize;


/**
 * All scalar function optimizers extend this class.
 * 
 * @author Nicholas Andrews
 */
public abstract class RealScalarFunctionOptimizer {
	RealScalarFunction f;
	public RealScalarFunctionOptimizer(RealScalarFunction f) {
		this.f = f;
	}
	public RealScalarFunction getFunction() { return f; }
	public abstract void optimize();
}
