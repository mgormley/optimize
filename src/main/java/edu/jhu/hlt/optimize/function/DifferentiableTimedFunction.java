package edu.jhu.hlt.optimize.function;

import edu.jhu.prim.vector.IntDoubleVector;

/**
 * A real scalar differentiable function with a time limit specified as side
 * information.
 * 
 * @author noandrews
 */
public interface DifferentiableTimedFunction extends TimedFunction, DifferentiableFunction {

    /**
     * Gets gradient of this function at the specified point, computed in the
     * given amount of time. Stochastic functions may use the allocated time to
     * better approximate the function value, for example.
     * 
     * @param point The point at which the function is evaluated.
     * @param seconds Time in seconds in which to evaluate the function.
     * @return The output gradient, a vector of partial derivatives.
     */
    IntDoubleVector getGradient(IntDoubleVector point, double seconds);

    /**
     * Gets the gradient and value at the specified point, computed in the given
     * amount of time.
     * 
     * @param point The point at which the function is evaluated.
     * @param seconds Time in seconds in which to evaluate the function.
     * @return The value and gradient.
     */
    ValueGradient getValueGradient(IntDoubleVector point, double seconds);

}
