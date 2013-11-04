package edu.jhu.hlt.optimize;

import org.apache.log4j.Logger;

import edu.jhu.hlt.optimize.function.DifferentiableFunction;

/**
 * 
 * Wei Xu: Towards Optimal One Pass Large Scale Learning with Averaged Stochastic Gradient Descent, Technical report, 2010.
 * ArXiv Link (arXiv:1107.2490v1)
 * 
 * @author noandrews
 */
public class ASGD extends    AbstractOptimizer<DifferentiableFunction>
				  implements Optimizer<DifferentiableFunction>,
							 Maximizer<DifferentiableFunction> {

    private static final Logger log = Logger.getLogger(ASGD.class);
	
	public static class ASGDParam {
		double a = 1d;
		double gamma_0 = 1d;
		double c = 0.51;
	}
	
	ASGDParam param;
	
	public ASGD(DifferentiableFunction f) {
		super(f);
		param = new ASGDParam();
	}
	
	public ASGD(DifferentiableFunction f, ASGDParam param) {
		super(f);
		this.param = param;
	}

	public boolean optimize(boolean minimize) {
		
		// TODO
		
		return true;
	}
	
	@Override
	public boolean maximize(DifferentiableFunction function, double[] point) {
		this.f = function;
		this.f.setPoint(point);
		return optimize(false);
	}

	@Override
	public boolean maximize() {
		return optimize(false);
	}

	@Override
	public boolean minimize(DifferentiableFunction function, double[] initial) {
		this.f = function;
		this.f.setPoint(initial);
		return optimize(true);
	}

	@Override
	public boolean minimize() {
		return optimize(true);
	}
	
}
