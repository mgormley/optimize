package edu.jhu.hlt.optimize.function;

import edu.jhu.prim.vector.IntDoubleVector;

/**
 * A real scalar function which is evaluated on a batch of examples.
 * 
 * @author mgormley
 */
public interface BatchFunction extends Function {
    
    /**
     * Gets value of this function at the specified point, computed on the given batch of examples.
     * 
     * @param point The point at which the function is evaluated.
     * @param batch A set of indices indicating the examples over which the gradient should be computed.
     * @return The value of the function at the point.
     */
    double getValue(IntDoubleVector point, int[] batch);

    /**
     * Gets the number of examples.
     */
    int getNumExamples();

}