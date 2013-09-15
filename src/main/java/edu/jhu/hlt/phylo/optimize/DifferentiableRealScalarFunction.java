package edu.jhu.hlt.phylo.optimize;

/**
 * @author Nicholas Andrews
 */
public interface DifferentiableRealScalarFunction extends RealScalarFunction {
	
	/**
	 * @param param	Function parameters
	 * @return 		Vector of approximate first-order derivatives
	 */
	double [] grad(double [] param);
	
	/**
	 * Evaluates the gradient at the current function parameters
	 * 
	 * @return 		Vector of approximate first-order derivatives
	 */
	double [] grad();
}
