package edu.jhu.hlt.optimize;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;

public class Franks implements DifferentiableFunction {
	int n;
	int order = 1; // 1st derivatives only
	double [] point;
	
	public Franks() {
		this.n = 1;
		point = new double[1];
	}
	
	DerivativeStructure AD_getValue(double [] point) {
		DerivativeStructure x = new DerivativeStructure(n, order, 0, point[0]);
		return x.subtract(1).multiply(x.add(1)).multiply(x);
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
