package edu.jhu.hlt.optimize.function;

import edu.jhu.prim.matrix.DoubleMatrix;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * Immutable carrier class for a value, a gradient, and a hessian.
 * 
 * @author mgormley
 */
public class ValueGradientHessian extends ValueGradient {

    private DoubleMatrix hessian;
    
    public ValueGradientHessian(double value, IntDoubleVector gradient, DoubleMatrix hessian) {
        super(value, gradient);
        this.hessian = hessian;
    }

    public DoubleMatrix getHessian() {
        return hessian;
    }    
    
}
