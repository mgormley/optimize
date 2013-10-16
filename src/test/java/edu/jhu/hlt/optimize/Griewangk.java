package edu.jhu.hlt.optimize;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;

/**
 * Test area is usually restricted to hyphercube −600 ≤ x_i ≤ 600, i = 1, ..., n. 
 * Its global minimum equal f(x) = 0 is obtainable for x_i = 0, i = 1, ..., n.
 * 
 * @author noandrews
 */
public class Griewangk implements DifferentiableFunction {

	int n;
	int order = 1; // 1st derivatives only
	double [] point;
	
	public Griewangk(int dimension) {
		this.n = dimension;
	}
	
	DerivativeStructure AD_getValue(double [] point) {
		
		DerivativeStructure [] x = new DerivativeStructure[n];
		for(int i=0; i<x.length; i++) {
			x[i] = new DerivativeStructure(n, order, i, point[i]);
		}
		
		DerivativeStructure lhs = new DerivativeStructure(n, order, 0);
		for(int i=0; i<n; i++) {
			lhs = lhs.add(x[i].pow(2));
		}
		lhs = lhs.multiply(1d/4000d);
		
		DerivativeStructure rhs = new DerivativeStructure(n, order, 1);
		for(int i=0; i<n; i++) {
			rhs = rhs.multiply( x[i].divide(Math.sqrt(i)).cos() );
		}
		
		return lhs.subtract(rhs).add(1d);
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
