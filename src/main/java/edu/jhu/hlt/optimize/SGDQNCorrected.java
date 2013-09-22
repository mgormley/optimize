package edu.jhu.hlt.optimize;

import org.apache.log4j.Logger;

/**
 * SGD-QN: Careful Quasi-Newton Stochastic Gradient Descent Antoine Bordes, Léon
 * Bottou, Patrick Gallinari; 10(Jul):1737--1754, 2009. JMLR.
 * 
 * Erratum: SGDQN is Less Careful than Expected Antoine Bordes, Léon Bottou,
 * Patrick Gallinari, Jonathan Chang, S. Alex Smith Journal of Machine Learning
 * Research 11 (2010) 2229-2240
 * 
 * TODO: This paper takes advantage of sparsity in the online updates. However,
 * the current implementation however uses dense updates. All occurences of
 * double [] need to be replaced with abstract vector operations that can then
 * use sparse math under the hood.
 * 
 * @author noandrews
 */
public class SGDQNCorrected extends    Optimizer<DifferentiableFunction>
                            implements Maximizer<DifferentiableFunction>, 
                                       Minimizer<DifferentiableFunction> {
	
    private static final Logger log = Logger.getLogger(SGDQNCorrected.class);
	
    public static class SGDQNCorrectedParam {
    	int T = 100;
    	double lambda = 1e-4;
    	double t0 = 1000;
    	int skip = 1;
    }
	
    SGDQNCorrectedParam param;
    
	// Work storage
	double [] prev_gradient;
	
	public SGDQNCorrected(DifferentiableFunction f, SGDQNCorrectedParam param) {
		super(f);
		this.param = param;
	}
	
	public SGDQNCorrected(DifferentiableFunction f, int T) {
		super(f);
		param = new SGDQNCorrectedParam();
		param.T = T;
	}
	
	public SGDQNCorrected(DifferentiableFunction f) {
		super(f);
		param = new SGDQNCorrectedParam();
	}

	public void updateFunction(int t, double [] B, boolean maximize) {
		double [] pt = f.getPoint();
		assert( pt != null );
		for(int i=0; i<pt.length; i++) {
			if(maximize) {
				pt[i] += 1.0/(t+param.t0)*B[i];
			} else {
				pt[i] -= 1.0/(t+param.t0)*B[i];
			}
		}
		f.setPoint(pt);
	}
	
	/**
	 * @param new_gradient   Gradient evaluated at the current time step (at new_pt)
	 * @param gradient       Gradient evaluated at the previous time step (at pt)
	 * @param newpt          Value of the variables at the current time step
	 * @param pt             Value of the variables at the previous time step
	 * @param skip           Number of iterations since the last Hessian approximation
	 * @param lambda         Weight decay parameter
	 * @return H		     Diagonal approximation to the Hessian
	 */
	public static void approxDiagHessian(double [] new_gradient, double [] gradient,
									     double [] new_pt, double [] pt,
									     double lambda,
									     int skip,
									     double [] H) {
		for(int i=0; i<pt.length; i++) {
			double r = Math.max(lambda, Math.min(100.0*lambda, new_gradient[i] - gradient[i]));
			H[i] = H[i] * 1.0/(1.0 - H[i]*r);
		}
	}
	
	public boolean optimize(boolean maximize) {
		int t = 0;
		int count = param.skip;
		boolean updateB = false;
		int r = 2;
		double [] B = new double[f.getNumDimensions()];
		for(int i=0; i<B.length; i++) {
			B[i] = param.lambda;
		}
		double [] p = new double[f.getNumDimensions()];
		
		double [] gradient     = new double[f.getNumDimensions()];
		double [] new_gradient = new double[f.getNumDimensions()];
		
		log.debug("starting optimization");
		
		while(t <= param.T) {
			
			if(updateB) {
				double [] pt = f.getPoint();
				f.getGradient(gradient);
				updateFunction(t, B, maximize);
				double [] new_pt = f.getPoint();
				f.getGradient(new_gradient);
				for(int i=0; i<p.length; i++) {
					p[i] = new_gradient[i] - gradient[i];
					B[i] += (2.0/r)*(new_pt[i]-pt[i])*(1.0/p[i]-B[i]);
					B[i] = Math.max(B[i], 1e-2*(1.0/param.lambda));
				}
				r += 1;
			} else {
				updateFunction(t, B, maximize);
			}
			count -= 1;
			if(count <= 0) {
				double [] pt = f.getPoint();
				for(int i=0; i<pt.length; i++) {
					pt[i] -= (double)param.skip/(double)(t+param.t0)*param.lambda*B[i];
				}
				count = param.skip;
				updateB = true;
			}
			t += 1;
			
			log.debug("ll @ iter " + t + " = " + f.getValue());
		}
		
		return true;
	}
	
	@Override
	public boolean minimize(DifferentiableFunction f, double[] initial) {
		return optimize(false);
	
	}

	@Override
	public boolean maximize(DifferentiableFunction function, double[] point) {
		return optimize(true);
	}

	@Override
	public boolean minimize() {
		return optimize(false);
	}

	@Override
	public boolean maximize() {
		return optimize(true);
	}
}
