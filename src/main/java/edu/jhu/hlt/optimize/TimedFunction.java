package edu.jhu.hlt.optimize;

/**
 * @author noandrews
 */
public interface TimedFunction extends Function {

    /**
     * Gets value of this function at the current point, computed in the given amount of time.
     * Stochastic functions may use the allocated time to better estimate the function value,
     * for example.
     * 
     * @param seconds	Time in seconds in which to evaluate the function.
     * @return 			The value of the function at the point.
     */
    double getValue(double seconds);
    
    double getTime();
	
}
