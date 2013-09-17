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
		double [] pt = f.getPoint();
		for(int i=0; i<pt.length; i++) {
			pt[i] -= 1.0/(t+t0)*B[i];
		}
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
		
		double [] gradient = new double[f.getNumDimensions()];
		double [] new_gradient = new double[f.getNumDimensions()];
		
		while(t <= T) {
			
			if(updateB) {
				double [] pt = f.getPoint();
				f.getGradient(gradient);
				updateFunction(t, B);
				double [] new_pt = f.getPoint();
				f.getGradient(new_gradient);
				for(int i=0; i<p.length; i++) {
					p[i] = new_gradient[i] - gradient[i];
					B[i] += (2.0/r)*(new_pt[i]-pt[i])*(1.0/p[i]-B[i]);
					B[i] = Math.max(B[i], 1e-2*(1.0/lambda));
				}
				r += 1;
			} else {
				updateFunction(t, B);
			}
			count -= 1;
			if(count <= 0) {
				double [] pt = f.getPoint();
				for(int i=0; i<pt.length; i++) {
					pt[i] -= (double)skip/(double)(t+t0)*lambda*B[i];
				}
				count = skip;
				updateB = true;
			}
			t += 1;
		}
		
		return true;
	}

	@Override
	public boolean maximize(DifferentiableFunction function, double[] point) {
		// TODO Auto-generated method stub
		return true;
	}

	@Override
	public boolean minimize() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean maximize() {
		// TODO Auto-generated method stub
		return false;
	}
}
