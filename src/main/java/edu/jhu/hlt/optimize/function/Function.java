package edu.jhu.hlt.optimize.function;

import edu.jhu.prim.vector.IntDoubleVector;

/**
 * An unconstrained real scalar function.
 * 
 * f: \mathcal{R}^n --> \mathcal{R}
 * 
 * @author mgormley
 * @author noandrews
 */
public interface Function {
    
    /**
     * Get the value of this function at the specified point.
     * @param point The point at which the function is evaluated.
     */
    double getValue(IntDoubleVector point);

    /**
     * Gets the number of dimensions of the domain of this function.
     * 
     * @return The domain's dimensionality.
     */
    int getNumDimensions();

}