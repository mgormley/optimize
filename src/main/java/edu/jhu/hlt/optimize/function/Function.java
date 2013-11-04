package edu.jhu.hlt.optimize.function;

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
     * Sets the current point for this function.
     * @param point The point.
     */
    void setPoint(double[] point);

    /**
     * @return Current setting of the free variables of the function
     */
    double [] getPoint();
    
    /**
     * Get the value of this function at the specified point (no side effects).
     * @param point The point at which the function is evaluated.
     */
    // TODO: Remove this method.
    @Deprecated    
    double getValue(double [] point);
    
    /**
     * The value of this function at the current point.
     * @return The value of the function.
     */
    double getValue();

    /**
     * Gets the number of dimensions of the domain of this function.
     * 
     * @return The domain's dimensionality.
     */
    int getNumDimensions();

}