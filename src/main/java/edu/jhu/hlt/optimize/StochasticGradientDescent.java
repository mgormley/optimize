package edu.jhu.hlt.optimize;


public class StochasticGradientDescent extends StochasticRealScalarFunctionOptimizer {
	
	double total_time;
	double a;     // initial step size
	double alpha; // decay \in (0.5,1]
	
	public StochasticGradientDescent(DifferentiableRealScalarFunction f, double t, double a, double alpha) {
		super(f);
		this.total_time = t;
		this.a = a;
		this.alpha = alpha;
	}

	@Override
	public void optimize() {
		double curr_t = 0d;
		int i = 0;
		DifferentiableRealScalarFunction fdif = (DifferentiableRealScalarFunction)f;
		do {
			long startTime = System.nanoTime();
			double [] g = fdif.grad();
			double [] param = fdif.get();
			double rate = a / Math.pow(1d+i, alpha);
			for(int k=0; k<fdif.dim(); k++) {
				param[k] = param[k] - rate*g[k];
			}
			i ++;
			curr_t += Util.nanoToSec(System.nanoTime() - startTime);
		} while(curr_t < total_time);
	}

	// FIXME: not using the argument
	@Override
	public double val(double t) {
		optimize();
		return f.val();
	}

	@Override
	public double val(double[] param, double t) {
		total_time = param[0];
		a          = param[1];
		alpha      = param[2];
		optimize();
		return f.val();
	}

	@Override
	public int dim() {
		return get().length;
	}

	@Override
	public void set(double[] param) {
		total_time = param[0];
		a = param[1];
		alpha = param[2];
	}

	@Override
	public double[] get() {
		return new double [] {total_time, a, alpha};
	}

	@Override
	public double val() {
		optimize();
		return f.val();
	}

	@Override
	public double val(double[] param) {
		throw new UnsupportedOperationException();
	}
}
