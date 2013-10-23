package edu.jhu.hlt.optimize.functions;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;

import edu.jhu.hlt.optimize.DifferentiableFunction;

/**
 * Test area is usually restricted to hyper-cube −500 ≤ x_i ≤ 500, i = 1, ..., n.
 * Its global minimum f(x) = −418.9829n is obtainable for x_i = 420.9687, i = 1, ..., n.
 * 
 * @author noandrews
 */
public class Schwefel implements DifferentiableFunction {

	int n;
	int order = 1; // 1st derivatives only
	double [] point;
	
	public Schwefel(int dimension) {
		this.n = dimension;
	}
	
	DerivativeStructure AD_getValue(double [] point) {
		
		DerivativeStructure [] x = new DerivativeStructure[n];
		for(int i=0; i<x.length; i++) {
			x[i] = new DerivativeStructure(n, order, i, point[i]);
		}
		
		DerivativeStructure value = new DerivativeStructure(n, order, 0);
		
		for(int i=0; i<n; i++) {
			value = value.add(x[i].abs().sqrt().multiply(x[i].negate()));
		}
		
		return value;
	}
	
	@Override
	public void setPoint(double[] point) {
		this.point = point;
		
	}

	@Override
	public double[] getPoint() {
		return point;
	}

	@Override
	public double getValue(double[] point) {
		return AD_getValue(point).getValue();
	}

	@Override
	public double getValue() {
		return getValue(point);
	}

	@Override
	public int getNumDimensions() {
		return n;
	}

	@Override
	public void getGradient(double[] gradient) {
		DerivativeStructure value = AD_getValue(point);
		for(int i=0; i<n; i++) {
			int [] orders = new int[n];
			orders[i] = 1;
			gradient[i] = value.getPartialDerivative(orders);
		}
	}

	
}
