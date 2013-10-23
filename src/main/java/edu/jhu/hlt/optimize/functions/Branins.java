package edu.jhu.hlt.optimize.functions;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;

import edu.jhu.hlt.optimize.DifferentiableFunction;

/**
 * Three global optima equal f(x1, x2) = 0.397887
 * 
 * 	(x1, x2) = (-pi, 12.275)
 * 	(x1, x2) = (+pi, 2.475)
 *  (x1, x2) = (9.42478, 2.475)
 *  
 * @author noandrews
 */
public class Branins implements DifferentiableFunction {

	static final double a = 1;
	static final double b = 5.1/4.0*Math.pow(Math.PI,2);
	static final double c = 5.0/Math.PI;
	static final double d = 6.0;
	static final double e = 10.0;
	static final double f = 1.0/8.0*Math.PI;
	
	int n;
	int order = 1; // 1st derivatives only
	double [] point;
	
	public Branins() {
		n = 3;
	}
	
	DerivativeStructure AD_getValue(double [] point) {
		
		DerivativeStructure [] x = new DerivativeStructure[n];
		for(int i=0; i<x.length; i++) {
			x[i] = new DerivativeStructure(n, order, i, point[i]);
		}
		
		return x[0].cos().multiply(e*(1-f)).add(x[1].subtract(x[0].pow(2).multiply(b)).add(x[0].multiply(c)).subtract(d).pow(2).multiply(a));		
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
