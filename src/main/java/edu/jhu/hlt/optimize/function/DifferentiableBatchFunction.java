package edu.jhu.hlt.optimize.function;

import edu.jhu.prim.vector.IntDoubleVector;

/**
 * A differentiable function, the gradient of which can be computed on a subset
 * of the examples.
 * 
 * @author mgormley
 * 
 */
public interface DifferentiableBatchFunction extends BatchFunction, DifferentiableFunction {

    /**
     * Adds the gradient at the specified point, computed on the given batch of
     * examples.
     * 
     * @param point The point at which the function is evaluated.
     * @param batch A set of indices indicating the examples over which the
     *            gradient should be computed.
     * @return The output gradient, a vector of partial derivatives.
     */
    IntDoubleVector getGradient(IntDoubleVector point, int[] batch);

    /**
     * Gets the gradient and value at the specified point, computed on the given
     * batch of examples.
     * 
     * @param point The point at which the function is evaluated.
     * @param batch A set of indices indicating the examples over which the
     *            gradient should be computed.
     * @return The value and gradient.
     */
    ValueGradient getValueGradient(IntDoubleVector point, int[] batch);

}
