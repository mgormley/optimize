package edu.jhu.hlt.optimize.linesearch;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;

import edu.jhu.hlt.optimize.GradientDescentWithLineSearch;
import edu.jhu.hlt.optimize.function.Bounds;
import edu.jhu.hlt.optimize.function.ConstrainedDifferentiableFunction;
import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.util.math.Vectors;
import edu.jhu.prim.util.Lambda.FnIntDoubleToDouble;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * This line search does some fancy things to ensure the returned step sizes
 * are robust.
 * 
 * Ref: http://terminus.sdsu.edu/SDSU/Math693a_f2013/Lectures/06/lecture.pdf
 * 
 * FIXME:
 * 	- Doesn't check bounds
 * 
 * @author noandrews
 */
public class ParanoidLineSearch {
	
	DifferentiableFunction f;
	boolean maximize;
	Bounds b;
	boolean checkBounds = false;
	ParanoidLineSearchPrm prm;
	
	double prev_alpha;
	double prev_dot;
	
    private static final Logger log = Logger.getLogger(GradientDescentWithLineSearch.class);
	
	public static class ParanoidLineSearchPrm {
		public double alpha_1 = 1.0;
		public double alpha_max = 1e2;
		public double c1 = 1e-4;
		public double c2 = 0.9;
		public double min_step_size = 1e-9;
		public double min_dot_0 = 1e-9;
	};
	
	public ParanoidLineSearch(DifferentiableFunction f) {
		this.f = f;
		if(f instanceof ConstrainedDifferentiableFunction) {
			b = ((ConstrainedDifferentiableFunction)f).getBounds();
			checkBounds = true;
		}
		prm = new ParanoidLineSearchPrm();
	}
	
	/**
	 * Find a step size which robustly decreases the given function.
	 * 
	 * @param f				Function to optimize
	 * @param maximize		If the function is being maximized or minimized
	 * @param x				The input point
	 * @param g				Gradient of the function evaluated at x
	 * @return				Step size
	 */
	public double search(IntDoubleVector x, IntDoubleVector g) {
    	
		List<Double> alphas = new ArrayList<Double>(); // list of previously set step sizes
		alphas.add(0.0);
		
		int i = 1;
    	double theta_0 = Theta(x, g, 0);
    	double theta_dot_0 = ThetaGrad(x, g, 0);
    	
    	// Check for convergence
    	//if(theta_dot_0 < prm.min_dot_0) {
    		//log.info("theta_dot_0 < eps; converged");
    		//return 0.0;
    	//}
    	
    	log.info("theta_0 = " + theta_0);
    	log.info("theta_dot_0 = " + theta_dot_0);
    	
    	double alpha;
    	//if(i==1) {
    	if(true) {
    		alpha = prm.alpha_1;
    	} else {
    		alpha = getInitialStep(theta_dot_0);
    	}
		
    	double theta_alpha = -1;
    	double theta_dot_alpha = -1;
        while(true) {
        	log.info("alpha = " + alpha);
        	theta_alpha = Theta(x, g, alpha);
        	log.info("theta alpha = " + theta_alpha);
        	if(theta_alpha > theta_0 + prm.c1*alpha*theta_dot_0) {
        		log.info("1");
        		alpha = zoom(alphas.get(i-1), alpha, theta_0, theta_dot_0, x, g);
        		break;
        	}
        	if(i > 1 && theta_alpha >= Theta(x, g, alphas.get(i-1))) {
        		log.info("2");
        		alpha = zoom(alphas.get(i-1), alpha, theta_0, theta_dot_0, x, g);
        		break;
        	}
        	theta_dot_alpha = ThetaGrad(x, g, alpha);
        	if(Math.abs(theta_dot_alpha) <= -prm.c2*theta_dot_0) {
        		log.info("3");
        		alpha = zoom(alpha, alphas.get(i-1), theta_0, theta_dot_0, x, g);
        		break;
        	}
        	if(theta_dot_alpha >= 0) {
        		log.info("4");
        		alpha = zoom(alpha, alphas.get(i-1), theta_0, theta_dot_0, x, g);
        		break;
        	}
        	// Increase step size
        	alpha *= 2;
        	if(alpha > prm.alpha_max) {
        		throw new RuntimeException("Failed to find a good step size");
        	}
            alphas.add(alpha);
        	i+=1;
        }
        
        prev_alpha = alpha;
        prev_dot = theta_dot_alpha; // FIXME: may not be set above
        
        log.info("picked step size = " + alpha);
        return alpha;
	}
	
	public boolean converged() {
		if(prev_dot <= prm.min_dot_0) {
			return true;
		}
		return false;
	}
	
	/**
	 * @param 	step
	 * @return	f(x+step*g)
	 */
	public double Theta(IntDoubleVector x, IntDoubleVector g, final double step) {
		
		// Scale the gradient by the learning rate.
        g.apply(new FnIntDoubleToDouble() {
            @Override
            public double call(int index, double value) {
            	return - step * value;
            }
        });
		
        IntDoubleVector new_x = x.copy();
        new_x.add(g);   	
    	double new_value = f.getValue(new_x);
		
		return new_value;
	}
	
	/**
	 * @param x		Initial point
	 * @param g		Gradient of f at x
	 * @param step  Step size
	 * @return		g dot f'(x+step*g)
	 */
	public double ThetaGrad(IntDoubleVector x, IntDoubleVector g, final double step) {
		
		// Scale the gradient by the learning rate.
        g.apply(new FnIntDoubleToDouble() {
            @Override
            public double call(int index, double value) {
            	return - step * value;
            }
        });
		
        IntDoubleVector new_x = x.copy();
        new_x.add(g);   	
		
		IntDoubleVector new_g = f.getGradient(new_x);
		return -new_g.dot(g);
	}
    
    // Assume rate of change in current iteration will be the same
    // as the previous iteration.
    public double getInitialStep(double curr_dot) {
    	return prev_alpha*prev_dot/curr_dot;
    }
    
    // Interpolate between alpha_min and alpha_max
    public double interpolate(double alpha_min, double alpha_max) {
    	return (alpha_min+alpha_max)/2;
    }
    
    public double zoom(double alpha_min, double alpha_max, 
    				   double theta_zero, double theta_dot_zero,
    		           IntDoubleVector x, IntDoubleVector g) {
    	
    	log.info("zoom: " + alpha_min + ", " + alpha_max);
    	
    	if(alpha_min == alpha_max) {
    		log.info("alpha_min = alpha_max");
    		return alpha_min;
    	}
    	    	
    	int j=0;
    	while(true) {
    		log.info("zoom: " + alpha_min + ", " + alpha_max);
    		if(alpha_max < prm.min_step_size) {
    			return 0;
    		}
    		double new_alpha = interpolate(alpha_min, alpha_max);
    		log.info("new_alpha = " + new_alpha);
    		assert(new_alpha <= alpha_max && new_alpha >= alpha_min) : "interpolation failed";
    		double theta_new_alpha = Theta(x, g, new_alpha);
    		
    		if(theta_new_alpha < prm.min_dot_0) {
    			return new_alpha;
    		}
    		
    		double theta_alpha_min = Theta(x, g, alpha_min);
    		log.info("theta_new_alpha = " + theta_new_alpha);
    		log.info("theta_alpha_min = " + theta_alpha_min);
    		if((theta_new_alpha > theta_zero + prm.c1*new_alpha*theta_dot_zero) ||
    				(theta_new_alpha >= theta_alpha_min)) {
    			log.info("0");
    			alpha_max = new_alpha;
    		} else {
    			double theta_dot_new_alpha = ThetaGrad(x, g, new_alpha);
    			if(Math.abs(theta_dot_new_alpha) <= -prm.c2*theta_dot_zero) {
    				log.info("1");
    				return new_alpha;
    			}
    			if(theta_dot_new_alpha*(alpha_max-alpha_min) >= 0) {
    				log.info("2");
    				alpha_max = alpha_min;
    			}
    			alpha_min = alpha_max; // FIXME: should this be out of the else?
    		}
    		j++;
    		//if(j>3) {
    		///	System.exit(1);
    		//}
    	}
    	
    }
}

