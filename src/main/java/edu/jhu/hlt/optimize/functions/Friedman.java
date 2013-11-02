package edu.jhu.hlt.optimize.functions;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;

import edu.jhu.hlt.optimize.DifferentiableFunction;
import edu.jhu.hlt.optimize.Function;

/**
 * 
 * @author Function used for regression tests
 *
 */
public class Friedman implements Function {
	
	int n;
	double [] point;
	
	public Friedman() {
		n = 5;
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
	public double getValue(double[] x) {
		return 10.0*Math.sin(Math.PI*x[0]*x[1])+20.0*(x[2]-0.5)+10.0*x[3]+5.0*x[4];
	}

	@Override
	public double getValue() {
		return getValue(point);
	}

	@Override
	public int getNumDimensions() {
		return n;
	}	
}