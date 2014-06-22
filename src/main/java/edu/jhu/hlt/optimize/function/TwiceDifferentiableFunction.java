package edu.jhu.hlt.optimize.function;

import edu.jhu.prim.matrix.DoubleMatrix;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * A real scalar differentiable function.
 * 
 * @author noandrews
 *
 */
public interface TwiceDifferentiableFunction extends DifferentiableFunction {
    
    /**
     * Gets the Hessian at the specified point.
     * 
     * @param point The point at which the function is evaluated.
     * @param H The output Hessian, a matrix of second derivatives.
     */
    void getHessian(IntDoubleVector point, DoubleMatrix H);

    /** 
     * Gets the value, gradient, and hessian at the specified point.
     * 
     * @param point The point at which the function is evaluated.
     * @return The value, gradient, and hessian output.
     */
    ValueGradientHessian getValueGradientHessian(IntDoubleVector point);
    
}