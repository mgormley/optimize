package edu.jhu.hlt.optimize.functions;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;

import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.prim.vector.IntDoubleDenseVector;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * Test area is usually restricted to hyper-cube −500 ≤ x_i ≤ 500, i = 1, ..., n.
 * Its global minimum f(x) = −418.9829n is obtainable for x_i = 420.9687, i = 1, ..., n.
 * 
 * @author noandrews
 */
public class Schwefel implements DifferentiableFunction {

	int n;
	int order = 1; // 1st derivatives only
	
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
	public double getValue(IntDoubleVector pt) {
		double [] x = new double[n];
		for(int i=0; i<n; i++) {
			x[i] = pt.get(i);
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
