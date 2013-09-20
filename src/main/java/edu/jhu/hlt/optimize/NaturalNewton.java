package edu.jhu.hlt.optimize;

public class NaturalNewton extends    Optimizer<DifferentiableFunction>
                           implements Maximizer<DifferentiableFunction>, 
                                      Minimizer<DifferentiableFunction> {

	public NaturalNewton(DifferentiableFunction f, double lambda) {
		super(f);
	}
	
	public NaturalNewton(DifferentiableFunction f) {
		super(f);
		// TODO Auto-generated constructor stub
	}

	@Override
	public boolean minimize(DifferentiableFunction function, double[] initial) {
		this.f = function;
		f.setPoint(initial);
		return this.minimize();
	}

	@Override
	public boolean minimize() {
		// TODO
		return true;
	}

	@Override
	public boolean maximize(DifferentiableFunction function, double[] point) {
		this.f = function;
		f.setPoint(point);
		return this.maximize();
	}

	@Override
	public boolean maximize() {
		// TODO
		return true;
	}
	

}
