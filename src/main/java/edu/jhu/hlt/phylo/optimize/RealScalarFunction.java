package edu.jhu.hlt.phylo.optimize;

/**
 * Interface for functions of many variables which evaluate to a single scalar
 * 
 * @author Nicholas Andrews
 */
public interface RealScalarFunction {
	
	/**
	 * @return Return the value of the function at the current value of its parameters
	 */
	double val();
	
	/**
	 * @param param	Function parameters
	 * @return 		Value of the function at param
	 */
	double val(double [] param);
	
	/**
	 * @param param Function parameters
	 */
	void set(double [] param);
	
	/**
	 * @return 		Function parameters
	 */
	double [] get();
	
	/**
	 * @return		The number of (free) variables
	 */
	double dim();
}
