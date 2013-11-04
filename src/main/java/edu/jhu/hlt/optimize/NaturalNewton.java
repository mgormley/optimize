package edu.jhu.hlt.optimize;

import org.apache.log4j.Logger;

import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.util.math.Vectors;

/**
 * A fast natural Newton method 
 * by Andrew Fitzgibbon and Nicolas Le Roux
 * ICML 2010
 *  
 * Nowadays, for many tasks such as object recognition or language
 * modeling, data is plentiful. As such, an important challenge has become to
 * find learning algorithms which can make use of all the available data. In
 * this setting, called ``large-scale learning'' by Bottou and Bousquet (2008),
 * learning and optimization become different and powerful optimization
 * algorithms are suboptimal learning algorithms. While most efforts are focused
 * on adapting optimization algorithms for learning by efficiently using the
 * information contained in the Hessian, Le Roux et al. (2008) exploited the
 * special structure of the learning problem to achieve faster convergence. In
 * this paper, we investigate a natural way of combining these two directions to
 * yield fast and robust learning algorithms.
 * 
 * Implementation notes
 *  - For the Hessian update, the algorithm from 
 *    
 *    	"Erratum: SGDQN is Less Careful than Expected", Bordes et al.
 *      Journal of Machine Learning Research 11 (2010) 2229-2240
 *      
 *    is used, which looks like a one-sided finite difference method.
 * 
 * @author noandrews
 */
public class NaturalNewton extends    Optimizer<DifferentiableFunction>
                           implements Maximizer<DifferentiableFunction>, 
                                      Minimizer<DifferentiableFunction> {

    private static final Logger log = Logger.getLogger(NaturalNewton.class);
	
	public static class NaturalNewtonParam {
		int skip  = 1;       // number of gradient updates between Hessian updates
		int skipC = 1;       // number of Hessian updates between covariance updates
		double gamma = 0.995; // discount factor for the moving covariance
		int T = 100;         // total number of iterations
		int t0 = 10;        // weight decay
		double lambda = 0.1; // weight decay
	}
	
	NaturalNewtonParam param;
	
	public NaturalNewton(DifferentiableFunction f, NaturalNewtonParam param) {
		super(f);
		this.param = param;
	}
	
	public NaturalNewton(DifferentiableFunction f) {
		super(f);
		param = new NaturalNewtonParam();
	}

	@Override
	public boolean minimize(DifferentiableFunction function, double[] initial) {
		this.f = function;
		f.setPoint(initial);
		return optimize(true);
	}
	
	public boolean optimize(boolean minimize) {
		
		int t = 0;
		int count = param.skip;
		int countC = param.skipC;
		
		// Storage for derivatives
		double [] theta      = new double [ f.getNumDimensions() ];
		double [] prev_theta = new double [ f.getNumDimensions() ];
		double [] g          = new double [ f.getNumDimensions() ];
		double [] prev_g     = new double [ f.getNumDimensions() ];
		double [] H          = new double [ f.getNumDimensions() ];
		double [] D          = new double [ f.getNumDimensions() ];
		double [] d          = new double [ f.getNumDimensions() ];
		
		// Storage for covariance updates
		double [] C          = new double [ f.getNumDimensions() ];	
		double gamma_t       = 0d;
		double delta_t       = 0d;
		double [] mu_t       = new double [ f.getNumDimensions() ];
		
		// Initialization
		for(int k=0; k<H.length; k++) {
			H[k] = 1.0/param.lambda;
			D[k] = H[k];
		}
		
		log.debug("starting optimization");
		
		while(t!=param.T) {
			
			log.debug("value @ iter " + t + " = " + f.getValue());
			
			f.getGradient(g);
			theta = f.getPoint();
			
			if(minimize) {
				for(int k=0; k<f.getNumDimensions(); k++) {
					theta[k] = theta[k] - 1d/(double)(t+param.t0)*D[k]*g[k];
				}
			} else {
				for(int k=0; k<f.getNumDimensions(); k++) {
					theta[k] = theta[k] + 1d/(double)(t+param.t0)*D[k]*g[k];
				}
			}
			f.setPoint(theta);
			
			if(count==0) {
				// Update the Hessian
				count = param.skip;
				
				SGDQNCorrected.approxDiagHessian(g, prev_g, 
											     theta, prev_theta, 
											     param.lambda, 
											     param.skipC, 
											     H);
				
				// Save the gradient and the point at which it was evaluated
				prev_g = g;
				prev_theta = theta;
				
				if(countC==0) {
					// Update the covariance
					countC = param.skipC;
					
					gamma_t = gamma_t*param.gamma + 1d;
					delta_t = delta_t*param.gamma*param.gamma + 1d;
					
					for(int k=0; k<theta.length; k++) {
						d[k] = 1d/H[k]*g[k];
						mu_t[k] = ((gamma_t - 1d)*mu_t[k] + d[k])  /(gamma_t);
						C[k]    = ((gamma_t - 1d)*C[k] + d[k]*d[k])/(gamma_t);
					}
					
					double N = 1d - (delta_t/(gamma_t*gamma_t));
					double l2 = Vectors.norm2(mu_t);
					
					for(int k=0; k<theta.length; k++) {
						D[k] = 1d/((1d + C[k] - mu_t[k]*mu_t[k])/(N*l2));
					}
				} else {
					countC --;
				}
			} else {
				count --;
			}
			
			t ++;
		}
		
		return true;
	}

	@Override
	public boolean maximize(DifferentiableFunction function, double[] point) {
		this.f = function;
		f.setPoint(point);
		return optimize(false);
	}

	@Override
	public boolean maximize() {
		return optimize(false);
	}

	@Override
	public boolean minimize() {
		return optimize(true);
	}
}
