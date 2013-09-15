package edu.jhu.hlt.phylo.optimize;

/**
 * @author Nicholas Andrews
 */
public interface DifferentiableStochasticRealScalarFunction extends StochasticRealScalarFunction {
	
	/**
	 * @param param	Function parameters
	 * @return 		Vector of (exact) first-order derivatives
	 */
	double [] grad(double [] param);
	
	/**
	 * Like gradient, this method returns the approximate value of the function
	 * at the given parameters. The additional parameter t specifies how 
	 * much compute time (in milliseconds) is available for this computation.
	 * Given more time, some functions may return more precise values.
	 * 
	 * @param param 	Function parameters
	 * @param millis	Time budget to evaluate the function gradient
	 * @return 			Noisy gradient evaluated at param
	 */
	double [] grad(double [] param, double t);
	
	double [] grad(double t);
	double [] grad();
}