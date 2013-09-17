package edu.jhu.hlt.optimize;

/**
 * A real scalar function.
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