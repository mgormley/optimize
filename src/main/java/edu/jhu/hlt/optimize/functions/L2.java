package edu.jhu.hlt.optimize.functions;

import edu.jhu.hlt.optimize.function.Regularizer;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.prim.vector.IntDoubleDenseVector;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * Gaussian prior (L2 regularizer) on the parameters with mean zero and a
 * specified variance.
 * 
 * @author mgormley
 * 
 */
public class L2 implements Regularizer {

    private double variance;
    private int numParams;
    
    public L2(double variance) {
        this.variance = variance;
    }
    
    /**
     * Builds a Gaussian prior (L2 regularizer) on the parameters.
     * 
     * @param variance The covariance matrix of the Gaussian will be variance*I.
     * @param numParams The number of parameters.
     */
    public L2(double variance, int numParams) {
        this.variance = variance;
        this.numParams = numParams;
    }
    
    /**
     * Gets the negated sum of squares times 1/(2\sigma^2).
     */
    @Override
    public double getValue(IntDoubleVector params) {
        double sum = params.dot(params);
        sum /= (2 * variance);
        return - sum;
    }

    /**
     * Gets the negative parameter value times 1/(\sigma^2).
     */
    // TODO: Why do Sutton & McCallum include the sum of the parameters here and not just the value for each term of the gradient.
    @Override
    public IntDoubleVector getGradient(IntDoubleVector params) {
        IntDoubleDenseVector gradient = new IntDoubleDenseVector(numParams);
        for (int j=0; j<numParams; j++) {
            gradient.set(j, - params.get(j) / variance);
        }
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
