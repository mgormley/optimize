package edu.jhu.hlt.optimize.function;

/**
 * A real scalar differentiable function.
 * 
 * @author noandrews
 *
 */
public interface TwiceDifferentiableFunction extends DifferentiableFunction {
    /**
     * Gets the Hessian at the given point
     * @param H The output Hessian, a matrix of second derivatives.
     */
    void getHessian(double[][] H);
}