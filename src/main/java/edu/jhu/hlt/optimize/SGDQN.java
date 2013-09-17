package edu.jhu.hlt.optimize;

/**
 *  SGD-QN: Careful Quasi-Newton Stochastic Gradient Descent
 *  Antoine Bordes, Léon Bottou, Patrick Gallinari; 10(Jul):1737--1754, 2009. JMLR.
 * 
 * @author noandrews
 */
public class SGDQN extends    Optimizer<DifferentiableFunction>
                   implements Maximizer<DifferentiableFunction>, 
                              Minimizer<DifferentiableFunction> {
	int T;
	double lambda;
	double t0;
	int skip;
	
	// Work storage
	double [] prev_gradient;
	
	public SGDQN(DifferentiableFunction f, double lambda, double t0, int T, int skip) {
		super(f);
		this.lambda = lambda;
		this.t0 = t0;
		this.T = T;
		this.skip = skip;
	}
	
	public SGDQN(DifferentiableFunction f, int T) {
		super(f);
		this.T = T;
		this.lambda = 1e-4;
		this.t0 = 1e4;
		this.skip = 16;
	}

	public void updateFunction(int t, double [] B) {
		double [] pt = ;
		f.setPoint(point);
	}
	
	@Override
	public boolean minimize(DifferentiableFunction f, double[] initial) {
		
		int t = 0;
		int count = skip;
		boolean updateB = false;
		int r = 2;
		double [] B = new double[f.getNumDimensions()];
		for(int i=0; i<B.length; i++) {
			B[i] = lambda;
		}
		
		double [] p = new double[f.getNumDimensions()];
		
		
		while(t <= T) {
			
			if(updateB) {
				double [] gradient = null;
				f.getGradient(gradient);
				
				// Update parameters
				
				double [] new_gradient = null;
				
			} else {
				
				// Update parameters
				
				
			}
			
		}
		
		return true;
	}

	@Override
	public boolean maximize(DifferentiableFunction function, double[] point) {
		// TODO Auto-generated method stub
		return true;
	}
}
