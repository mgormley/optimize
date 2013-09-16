package edu.jhu.hlt.optimize;

/**
 * @author noandrews
 */
public interface DifferentiableTimedFunction extends TimedFunction, DifferentiableFunction {

	/**
     * Gets gradient of this function at the current point, computed in the given amount of time.
     * Stochastic functions may use the allocated time to better approximate the function value,
     * for example.
     * 
     * @param seconds	Time in seconds in which to evaluate the function.
     * @return 			The value of the function at the point.
     */
	void getGradient(double seconds, double [] gradient);
	
}
