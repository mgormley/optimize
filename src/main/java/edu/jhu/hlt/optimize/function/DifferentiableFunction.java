package edu.jhu.hlt.optimize.function;

import edu.jhu.prim.vector.IntDoubleVector;

/**
 * A real scalar differentiable function.
 * 
 * @author mgormley
 *
 */
public interface DifferentiableFunction extends Function {

    /**
     * Gets the gradient at the specified point.
     * 
     * @param point The point at which the function is evaluated.
     * @return The output gradient, a vector of partial derivatives.
     */
    IntDoubleVector getGradient(IntDoubleVector point);
        
    /** 
     * Gets the gradient and value at the specified point.
     * 
     * @param point The point at which the function is evaluated.
     * @return The value and gradient.
     */
    ValueGradient getValueGradient(IntDoubleVector point);
    
}
