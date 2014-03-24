package edu.jhu.hlt.optimize.functions;

import edu.jhu.hlt.optimize.function.Function;
import edu.jhu.prim.vector.IntDoubleVector;

public class Line implements Function {

	double slope;
	double intercept;
	
	public Line(double slope, double intercept) {
		this.slope = slope;
		this.intercept = intercept;
	}
	
	public double getValue(double x) {
		return x*slope + intercept;
	}
	
	@Override
	public double getValue(IntDoubleVector point) {
		return point.get(0)*slope + intercept;
	}

	@Override
	public int getNumDimensions() {
		return 2;
	}
	
}
