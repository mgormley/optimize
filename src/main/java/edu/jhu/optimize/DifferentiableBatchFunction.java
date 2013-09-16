package edu.jhu.optimize;


/**
 * A differentiable function, the gradient of which can be computed on a subset of the examples.
 * 
 * @author mgormley
 *
 */
public interface DifferentiableBatchFunction extends BatchFunction, DifferentiableFunction {

    /**
     * Adds the gradient at the current point, computed on the given batch of examples.
     * @param batch A set of indices indicating the examples over which the gradient should be computed.
     * @param gradient The output gradient, a vector of partial derivatives.
     */
    void getGradient(int[] batch, double[] gradient);
    
}

