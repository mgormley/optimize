package edu.jhu.hlt.optimize.function;

import edu.jhu.prim.vector.IntDoubleVector;

/**
 * A real scalar differentiable function with a time limit specified as side information.
 * 
 * @author noandrews
 * @author mgormley
 */
public interface TimedFunction extends Function {

    /**
     * Gets value of this function at the current point, computed in the given amount of time.
     * Function approximators may use the allocated time to produce better estimates.
     * 
     * @param point The point at which the function is evaluated.
     * @param seconds	Time in seconds in which to evaluate the function.
     * @return 			The value of the function at the point.
     */
    double getValue(IntDoubleVector point, double seconds);
    
    double getTime();
	
}
