package edu.jhu.hlt.optimize.functions;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;

import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.prim.vector.IntDoubleDenseVector;
import edu.jhu.prim.vector.IntDoubleVector;

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
		DerivativeStructure value = AD_getValue(point);
		double [] gradient = new double[n];
		for(int i=0; i<n; i++) {
			int [] orders = new int[n];
			orders[i] = 1;
			gradient[i] = value.getPartialDerivative(orders);
		}
		return new IntDoubleDenseVector(gradient);
	}

	@Override
	public ValueGradient getValueGradient(IntDoubleVector point) {
		return new ValueGradient(getValue(point), getGradient(point));
	}
	
}
