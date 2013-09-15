package edu.jhu.hlt.phylo.optimize;

/**
 * All scalar function optimizers extend this class.
 * 
 * @author Nicholas Andrews
 */
public abstract class RealScalarFunctionOptimizer implements POMDP {
	RealScalarFunction f;
	public RealScalarFunctionOptimizer(RealScalarFunction f) {
		this.f = f;
	}
	
	public abstract void optimize();
}
