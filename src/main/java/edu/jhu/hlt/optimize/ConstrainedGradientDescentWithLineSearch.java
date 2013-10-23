package edu.jhu.hlt.optimize;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.interpolation.NevilleInterpolator;
import org.apache.commons.math3.analysis.interpolation.SplineInterpolator;
import org.apache.commons.math3.analysis.interpolation.UnivariateInterpolator;
import org.apache.log4j.Logger;

import edu.jhu.hlt.util.math.Vectors;

/**
 * Constrained gradient descent. A line search is used at each iteration to pick the step size.
 * The line search also ensures the step size does not result in any violated constraints.
 * 
 * FIXME:
 * 	- This has a lot in common with GradientDescentWithLineSearch. This one should
 *    be a subclass instead to avoid duplicate code.
 * 
 * @author noandrews
 */
public class ConstrainedGradientDescentWithLineSearch implements Maximizer<ConstrainedDifferentiableFunction>, 
                                                                 Minimizer<ConstrainedDifferentiableFunction> {

    /** Options for this optimizer. */
    public static class ConstrainedGradientDescentWithLineSearchPrm {
        /** The number of iterations to perform. */
        public int iterations = 10;
        
        // Magic linesearch parameters
        public int max_linesearch_iter = 50;
        public double initial_step = 0.001;
        public double tau = 0.5;
        public double c1 = 0.5;
        public double c2 = 1e-4;
        public double min_step_size = 1e-12;
        public double min_grad_norm = 1e-12; // TODO: unused
        
        public ConstrainedGradientDescentWithLineSearchPrm() { } 
        public ConstrainedGradientDescentWithLineSearchPrm(int iterations) {
            this.iterations = iterations;
        }
    }
    
    private static final Logger log = Logger.getLogger(ConstrainedGradientDescentWithLineSearch.class);

    private ConstrainedGradientDescentWithLineSearchPrm prm;

    // Temporary storage for interpolation of the step sizes
    List<Double> t;
    List<Double> alpha;
    
    public ConstrainedGradientDescentWithLineSearch(int iterations) {
        this(new ConstrainedGradientDescentWithLineSearchPrm(iterations));
    }
    
    public ConstrainedGradientDescentWithLineSearch(ConstrainedGradientDescentWithLineSearchPrm prm) {
        this.prm = prm;
    }

    /**
     * Maximize the function starting at the given initial point.
     */
    @Override
    public boolean maximize(ConstrainedDifferentiableFunction function, double[] point) {
        return optimize(function, point, true);
    }

    /**
     * Minimize the function starting at the given initial point.
     */
    public boolean minimize(ConstrainedDifferentiableFunction function, double[] point) {
        return optimize(function, point, false);
    }

    private boolean optimize(ConstrainedDifferentiableFunction function, double[] point, final boolean maximize) {        
        assert (function.getNumDimensions() == point.length);
        double[] gradient = new double[point.length];
        
        // Initialize work storage
        this.t = new ArrayList<Double>();
        this.alpha = new ArrayList<Double>();
        
        for (int iterCount=0; iterCount < prm.iterations; iterCount++) {
            function.setPoint(point);
            
            // Get the current value of the function.
            double value = function.getValue();
            //log.info(String.format("[iter %d] f(%f) = %f", iterCount, point[0], value));
            
            // Get the gradient of the function.
            Arrays.fill(gradient, 0.0);
            function.getGradient(gradient);
            assert (gradient.length == point.length);
            
            // Take a step in the direction of the gradient.
            double lr = lineSearch(function, maximize, point, gradient, iterCount);
            t.add((double)iterCount);
            alpha.add(lr);
            //log.info(String.format("[iter %d] using step = %f", iterCount, lr));
            
            if(lr < 0) {
            	log.info("Line search failed; stopping early");
            	break;
            }
            
            for (int i=0; i<point.length; i++) {
                if (maximize) {
                    point[i] += lr * gradient[i];
                } else {
                    point[i] -= lr * gradient[i];
                }
            }
        }
        
        // Get the final value of the function on all the examples.
        double value = function.getValue();
        log.info(String.format("Final function value = %g", value));
        
        // We don't test for convergence.
        return false;
    }

    // Armijo rule: ensures that the step size decreases f 'sufficiently'
    public static boolean sufficientDecrease(double new_value, double value, double step, double c1, double [] d, double [] g, boolean maximize) {      
    	if(maximize) {
    		if(new_value < value) return false;
        	return new_value >= value+c1*step*Vectors.dotProduct(d, g);
        } else {
        	if(new_value > value) return false;
        	return new_value <= value+c1*step*Vectors.dotProduct(d, g);
        }
    }

    // Curvature rule: ensures that the slope has been reduced 'sufficiently'
    public static boolean sufficientCurvature(double [] d, double [] g, double [] new_g, double c2, boolean maximize) {
    	return true;
//    	if(maximize) {
//    		return Math.abs(Vectors.dotProduct(d, new_g)) >= c2*Math.abs(Vectors.dotProduct(d, g));
//    	} else {
//    		return Math.abs(Vectors.dotProduct(d, new_g)) <= c2*Math.abs(Vectors.dotProduct(d, g));
//    	}
    }
    
    public static boolean satisfiesConstraints(ConstrainedDifferentiableFunction f, double [] new_pt) {
    	Bounds bounds = f.getBounds();
    	for(int i=0; i<f.getNumDimensions(); i++) {
    		if(new_pt[i] > bounds.getUpper(i) ||
    		   new_pt[i] < bounds.getLower(i)) {
    			return false;
    		}
    	}
    	return true;
    }

    private double lineSearch(ConstrainedDifferentiableFunction f, boolean maximize, double [] x, double [] g, int iterCount) {
    	
    	double step;
    	
    	if(true) {
    		step = prm.initial_step;
    	} 
    	
//    	else {
//    		UnivariateInterpolator interpolator = new NevilleInterpolator(); // overkill
//    		double [] xs = ArrayUtils.toPrimitive(t.toArray(new Double[t.size()]));
//    		double [] ys = ArrayUtils.toPrimitive(alpha.toArray(new Double[alpha.size()]));
//    		UnivariateFunction alpha_curve = interpolator.interpolate(xs, ys);
//    		step = alpha_curve.value(iterCount);
//    		log.info("predicted step size = " + step);
//    		if(step<0) {
//    			step = alpha.get(iterCount-1);
//    		}
//    		
//    	}
    	
        int iter = 0;
        double [] d = g;
        double value = f.getValue();
        double [] new_g = new double[f.getNumDimensions()];
        double [] new_x = new double[f.getNumDimensions()];
        boolean decrease;
		boolean curvature;
		boolean constraints;
				
		do {                        
        	for(int i=0; i<f.getNumDimensions(); i++) {
        		if(maximize){
        			new_x[i] = x[i] + step*g[i];
        		} else {
        			new_x[i] = x[i] - step*g[i];
        		}
        	}
        	f.setPoint(new_x);
        	double new_value = f.getValue();
        	f.getGradient(new_g);
        	
        	curvature = sufficientCurvature(d, g, new_g, prm.c2, maximize);
        	decrease  = sufficientDecrease(new_value, value, step, prm.c1, d, g, maximize);
        	constraints = satisfiesConstraints(f, new_x);
        	        	
        	if(curvature && decrease && constraints) {
        		return step;
        	}
        	
        	// Contract the step size
        	step *= prm.tau;        	
        	iter ++;
        	
        	if(iter >= prm.max_linesearch_iter) {
        		return -1;
        	}
        	
        	if(step < prm.min_step_size) {
        		return -1;
        	}
        } while(!curvature || !decrease || !constraints); 
 
        return step;
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