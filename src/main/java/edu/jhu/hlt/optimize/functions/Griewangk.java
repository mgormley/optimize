package edu.jhu.hlt.optimize.functions;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;

import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.prim.vector.IntDoubleDenseVector;
import edu.jhu.prim.vector.IntDoubleVector;

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
