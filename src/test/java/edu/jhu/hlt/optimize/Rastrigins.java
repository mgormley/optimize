package edu.jhu.hlt.optimize;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;

/**
 * Test area is usually restricted to hyper-cube −5.12 ≤ xi ≤ 5.12, i = 1, ..., n. 
 * Its global minimum equal f(x) = 0 is obtainable for x_i = 0 for i = 1, ..., n.
 * 
 * @author noandrews
 */
public class Rastrigins implements DifferentiableFunction {

	int n;
	int order = 1; // 1st derivatives only
	double [] point;
	
	public Rastrigins(int dimension) {
		this.n = dimension;
	}
	
	DerivativeStructure AD_getValue(double [] point) {
		
		DerivativeStructure [] x = new DerivativeStructure[n];
		for(int i=0; i<x.length; i++) {
			x[i] = new DerivativeStructure(n, order, i, point[i]);
		}
		
		DerivativeStructure value = new DerivativeStructure(n, order, 10d*n);
		
		for(int i=0; i<n; i++) {
			value = value.add(x[i].pow(2).subtract(x[i].multiply(2*Math.PI).cos().multiply(10)));
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
