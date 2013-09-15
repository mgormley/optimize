package edu.jhu.hlt.phylo.optimize;

/**
 * Interface for multi-variate functions whose value is approximate 
 * 
 * @author Nicholas Andrews
 */
public interface StochasticRealScalarFunction extends RealScalarFunction {
	/**
	 * Like value, this method returns the approximate value of the function
	 * at the given parameters. The additional parameter t specifies how 
	 * much compute time (in milliseconds) is available for this computation.
	 * Given more time, some functions may return more precise values of the
	 * function.
	 * 
	 * @param param Function parameters
	 * @param t     Time budget to evaluate the function value.
	 * @return 		Noisy value of the function evaluated at param
	 */
	double val(double [] param, double t);
	double val(double t);
}