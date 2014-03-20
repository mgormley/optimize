package edu.jhu.hlt.optimize.functions;

import java.util.Arrays;
import java.util.List;

import edu.jhu.hlt.optimize.function.Regularizer;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.prim.vector.IntDoubleDenseVector;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * L2 penalty where you can't have different means and variances
 * for each dimension (but no covariance).
 * 
 * @author travis
 */
public class HeterogeneousL2 implements Regularizer {
	
	public static HeterogeneousL2 zeroMeanIgnoringIndices(List<Integer> indicesNotToRegularize, double varianceForOthers, int dimension) {
		double[] means = new double[dimension];
		double[] variances = new double[dimension];
		Arrays.fill(variances, varianceForOthers);
		// 3 orders of magnitude is close enough to "not regularizing"
		for(Integer i : indicesNotToRegularize) {
			variances[i] = 1000d * varianceForOthers;
		}
		return new HeterogeneousL2(means, variances);
	}

	private double[] means;
	private double[] variances;
    private int numParams;
    
    public HeterogeneousL2(double[] means, double[] variances) {
    	if(means.length != variances.length)
    		throw new IllegalArgumentException();
    	for(int i=0; i<variances.length; i++) {
    		if(variances[i] <= 0d) {
    			throw new IllegalArgumentException();
    		}
    	}
    	this.means = means;
    	this.variances = variances;
    	this.numParams = means.length;
    }

	@Override
	public double getValue(IntDoubleVector point) {
		int n = means.length;
		double sum = 0d;
		for(int i=0; i<n; i++) {
			double d = point.get(i) - means[i];
			sum += d * d / (2d * variances[i]);
		}
		return -sum;
	}

	@Override
	public IntDoubleVector getGradient(IntDoubleVector point) {
	    IntDoubleVector gradient = new IntDoubleDenseVector(numParams);
		int n = means.length;
		for(int i=0; i<n; i++) {
			gradient.set(i, - (point.get(i) - means[i]) / variances[i]);
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

	@Override
	public void setNumDimensions(int numParams) {
		this.numParams = numParams;
	}

}
