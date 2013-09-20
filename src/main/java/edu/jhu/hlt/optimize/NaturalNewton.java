package edu.jhu.hlt.optimize;

public class NaturalNewton extends    Optimizer<DifferentiableFunction>
                           implements Maximizer<DifferentiableFunction>, 
                                      Minimizer<DifferentiableFunction> {

	public static class NaturalNewtonParam {
		int skip  = 8;       // number of gradient updates between Hessian updates
		int skipC = 8;       // number of Hessian updates between covariance updates
		double gamma;        // discount factor for the moving covariance
		int T = 100;         // total number of epochs
		int t0 = 100;        // weight decay
		double lambda = 1.0; // weight decay
	}
	
	NaturalNewtonParam param;
	
	public NaturalNewton(DifferentiableFunction f, NaturalNewtonParam param) {
		super(f);
		this.param = param;
	}
	
	public NaturalNewton(DifferentiableFunction f) {
		super(f);
	}

	@Override
	public boolean minimize(DifferentiableFunction function, double[] initial) {
		this.f = function;
		f.setPoint(initial);
		return this.minimize();
	}

	@Override
	public boolean minimize() {
		
		int t = 0;
		int count = param.skip;
		int countC = param.skipC;
		double gamma_0 = 0d;
		double delta_0 = 0d;
		double [] theta = new double [f.getNumDimensions()];
		double [] g     = new double [f.getNumDimensions()];
		double [] H     = new double [f.getNumDimensions()];
		double [] D     = new double [f.getNumDimensions()];
		double [] mu    = new double [f.getNumDimensions()];
		double [] C     = new double [f.getNumDimensions()];
		while(t!=param.T) {
			
			f.getGradient(g);
			theta = f.getPoint();
			for(int k=0; k<f.getNumDimensions(); k++) {
				theta[k] = theta[k] - 1d/(double)(t+param.t0)*D[k]*g[k];
			}
			f.setPoint(theta);
			
			if(count==0) {
				count = param.skip;
			}
						
		}
		
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
