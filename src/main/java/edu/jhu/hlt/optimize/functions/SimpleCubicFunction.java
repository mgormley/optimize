package edu.jhu.hlt.optimize.functions;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;

import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.prim.vector.IntDoubleDenseVector;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * The function f(x) = x(x-1)(x+1).
 */
public class SimpleCubicFunction implements DifferentiableFunction {
	int n;
	int order = 1; // 1st derivatives only
	
	public SimpleCubicFunction() {
		this.n = 1;
	}
	
	DerivativeStructure AD_getValue(double [] point) {
		DerivativeStructure x = new DerivativeStructure(n, order, 0, point[0]);
		return x.subtract(1).multiply(x.add(1)).multiply(x);
	}

	@Override
	public double getValue(IntDoubleVector point) {
		double [] x = new double[n];
		for(int i=0; i<n; i++) {
			x[i] = point.get(i);
		}
		return AD_getValue(x).getValue();
	}

	@Override
	public int getNumDimensions() {
		return n;
	}
	
	@Override
	public IntDoubleVector getGradient(IntDoubleVector pt) {
		double [] x = new double[n];
		double [] g = new double[n];
		for(int i=0; i<n; i++) {
			x[i] = pt.get(i);
		}
		DerivativeStructure value = AD_getValue(x);
		for(int i=0; i<n; i++) {
			int [] orders = new int[n];
			orders[i] = 1;
			g[i] = value.getPartialDerivative(orders);
		}
		return new IntDoubleDenseVector(g);
	}

	@Override
	public ValueGradient getValueGradient(IntDoubleVector point) {
		return new ValueGradient(getValue(point), getGradient(point));
	}

	
}
