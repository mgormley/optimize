package edu.jhu.hlt.optimize;

/**
 * A real scalar differentiable function.
 * 
 * @author mgormley
 *
 */
public interface DifferentiableFunction extends Function {

    /**
     * Gets the gradient at the current point.
     * @param gradient The output gradient, a vector of partial derivatives.
     */
    void getGradient(double[] gradient);
    
}
