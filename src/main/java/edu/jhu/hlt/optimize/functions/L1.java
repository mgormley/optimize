package edu.jhu.hlt.optimize.functions;

import edu.jhu.hlt.optimize.function.Regularizer;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.prim.vector.IntDoubleDenseVector;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * L1 regularizer on the parameters.
 * 
 * <p>
 * Of course, the L1 regularizer isn't actually differentiable at 0. While L1
 * regularization can be written out as a convex quadratic program and solved
 * with a generic solver, we instead simply return 0 for the case of the
 * gradient at 0 as a simple hack.
 * </p>
 * 
 * @author mgormley
 */
public class L1 implements Regularizer {

    private double lambda;
    private int numParams;
    
    public L1(double lambda) {
        this.lambda = lambda;
    }
    
    /**
     * Builds an L1 regularizer on the parameters.
     * 
     * @param lambda The multiplier on the L1 regularization term.
     * @param numParams The number of parameters.
     */
    public L1(double lambda, int numParams) {
        this.lambda = lambda;
        this.numParams = numParams;
    }
    
    /**
     * Gets - \lambda * |\theta|_1.
     */
    @Override
    public double getValue(IntDoubleVector params) {
        double sum = 0.0;
        for (int i=0; i<numParams; i++) {
            sum += Math.abs(params.get(i));
        }
        return - lambda * sum;
    }

    @Override
    public IntDoubleVector getGradient(IntDoubleVector params) {
        IntDoubleDenseVector gradient = new IntDoubleDenseVector(numParams);
        for (int j=0; j<numParams; j++) {
            if (params.get(j) < 0) {
                gradient.set(j, - lambda);
            } else if (params.get(j) > 0) {
                gradient.set(j, lambda);
            } else {
                // This is just a hack to work around the fact that L1 is not
                // differentiable at zero.
                gradient.set(j, 0);
            }
        }
        // Since we're subtracting this norm.
        gradient.scale(-1);
        return gradient;
    }

    @Override
    public ValueGradient getValueGradient(IntDoubleVector point) {
        return new ValueGradient(getValue(point), getGradient(point));
    }

    @Override
    public int getNumDimensions() {
        return numParams;
    }
    
    public void setNumDimensions(int numParams) {
        this.numParams = numParams ;
    }

}
