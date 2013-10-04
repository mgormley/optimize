package edu.jhu.hlt.optimize;

/** The function x^2. */
public class XSquared implements DifferentiableFunction {

	double x;
    
	/** Constructor with 0 as the initial point. */
	public XSquared() {
        this.x = 0;
    }
	   
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

	
}
