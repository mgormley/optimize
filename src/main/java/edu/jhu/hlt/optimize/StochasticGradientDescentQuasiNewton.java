package edu.jhu.hlt.optimize;

/**
 *  SGD-QN: Careful Quasi-Newton Stochastic Gradient Descent
 *  Antoine Bordes, Léon Bottou, Patrick Gallinari; 10(Jul):1737--1754, 2009. JMLR.
 * 
 * @author noandrews
 */
public class StochasticGradientDescentQuasiNewton extends    Optimizer<DifferentiableFunction>
                                                  implements Maximizer<DifferentiableFunction>, 
															 Minimizer<DifferentiableFunction>,
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
	public boolean minimize(DifferentiableFunction function,
			double[] initial) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean maximize(DifferentiableFunction function, double[] point) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void setPoint(double[] point) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double getValue() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public int getNumDimensions() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double getValue(double seconds) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double getTime() {
		// TODO Auto-generated method stub
		return 0;
	}
}
