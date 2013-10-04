package edu.jhu.hlt.optimize;

public class XSquared implements DifferentiableFunction {

	double x;
	
	public XSquared(double x) {
		this.x = x;
	}
	
	@Override
	public void setPoint(double[] point) {
		x = point[0];
	}

	@Override
	public double[] getPoint() {
		return new double [] {x};
	}

	@Override
	public double getValue() {
		return x*x;
	}

	@Override
	public int getNumDimensions() {
		return getPoint().length;
	}

	@Override
	public void getGradient(double[] gradient) {
		gradient[0] = 2*x;
	}

	@Override
	public double getValue(double[] point) {
		return point[0]*point[0];
	}

	
}
