package edu.jhu.hlt.optimize.functions;

import edu.jhu.hlt.optimize.function.Function;
import edu.jhu.prim.vector.IntDoubleVector;

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
	public double getValue(IntDoubleVector x) {
		return 10.0*Math.sin(Math.PI*x.get(0)*x.get(1))+20.0*(x.get(2)-0.5)+10.0*x.get(3)+5.0*x.get(4);
	}

	@Override
	public int getNumDimensions() {
		return n;
	}	
}