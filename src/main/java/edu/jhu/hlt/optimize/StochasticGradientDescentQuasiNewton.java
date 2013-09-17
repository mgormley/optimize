package edu.jhu.hlt.optimize;

/**
 *  SGD-QN: Careful Quasi-Newton Stochastic Gradient Descent
 *  Antoine Bordes, Léon Bottou, Patrick Gallinari; 10(Jul):1737--1754, 2009. JMLR.
 * 
 * @author noandrews
 */
public class StochasticGradientDescentQuasiNewton extends    Optimizer<DifferentiableFunction>
                                                  implements Minimizer<DifferentiableFunction>,
															 TimedFunction {
	
	double T;
	double lambda;
	double t0;
	int skip;
	
	public StochasticGradientDescentQuasiNewton(DifferentiableFunction f, double lambda, double t0, double T, int skip) {
		super(f);
		this.lambda = lambda;
		this.t0 = t0;
		this.T = T;
		this.skip = skip;
	}
	
	public StochasticGradientDescentQuasiNewton(DifferentiableFunction f, double T) {
		super(f);
		this.T = T;
		this.lambda = 1e-4;
		this.t0 = 1e4;
		this.skip = 16;
	}

	@Override
	public boolean minimize() {
		// TODO Auto-generated method stub
		return false;
	}
	
	@Override
	public boolean minimize(DifferentiableFunction function, double[] initial) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void setPoint(double[] point) {
		this.T      = point[0];
		this.lambda = point[1];
		this.t0     = point[2];
		this.skip   = (int)point[3];
	}

	@Override
	public double getValue() {
		minimize();
		return f.getValue();
	}

	@Override
	public int getNumDimensions() {
		return getPoint().length;
	}

	@Override
	public double getValue(double seconds) {
		this.T = seconds;
		minimize();
		return f.getValue();
	}

	@Override
	public double getTime() {
		return T;
	}

	@Override
	public double[] getPoint() {
		return new double [] {T, lambda, t0, skip};
	}
}
