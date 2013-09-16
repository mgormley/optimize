package edu.jhu.hlt.optimize;

public class XSquared implements DifferentiableRealScalarFunction {

	double x;
	
	public XSquared(double x) {
		this.x = x;
	}
	
	@Override
	public double val() {
		return x*x;
	}

	@Override
	public double val(double[] param) {
		return param[0]*param[0];
	}

	@Override
	public void set(double[] param) {
		x = param[0];
	}

	@Override
	public double[] get() {
		return new double[] {x};
	}

	@Override
	public int dim() {
		return 1;
	}

	@Override
	public double[] grad(double[] param) {
		return new double [] {2d*param[0]};
	}

	@Override
	public double[] grad() {
		return new double [] {2d*x};
	}
	
}
