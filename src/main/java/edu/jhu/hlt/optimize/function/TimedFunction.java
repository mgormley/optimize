package edu.jhu.hlt.optimize.function;

/**
 * @author noandrews
 */
public interface TimedFunction extends Function {

    /**
     * Gets value of this function at the current point, computed in the given amount of time.
     * Function approximators may use the allocated time to produce better estimates.
     * 
     * @param seconds	Time in seconds in which to evaluate the function.
     * @return 			The value of the function at the point.
     */
    double getValue(double seconds);
    
    double getTime();
	
}
