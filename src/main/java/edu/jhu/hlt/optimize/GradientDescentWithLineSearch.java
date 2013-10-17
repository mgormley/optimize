package edu.jhu.hlt.optimize;

import java.util.Arrays;

import org.apache.log4j.Logger;

import edu.jhu.hlt.util.math.Vectors;

/**
 * Gradient descent. A line search is used at each iteration to pick the step size.
 * 
 * @author noandrews
 */
public class GradientDescentWithLineSearch implements Maximizer<DifferentiableFunction>, 
                                                      Minimizer<DifferentiableFunction> {

    /** Options for this optimizer. */
    public static class GradientDescentWithLineSearchPrm {
        /** The number of iterations to perform. */
        public int iterations = 10;
        
        // Magic linesearch parameters
        public int max_linesearch_iter = 100;
        public double initial_step = 1.0;
        public double tau = 0.5;
        public double c1 = 0.5;
        public double c2 = 1e-4;
        
        public GradientDescentWithLineSearchPrm() { } 
        public GradientDescentWithLineSearchPrm(int iterations) {
            this.iterations = iterations;
        }
    }
    
    private static final Logger log = Logger.getLogger(GradientDescentWithLineSearch.class);

    /** The number of iterations performed thus far. */
    private int iterCount;

    private GradientDescentWithLineSearchPrm prm;
    
    public GradientDescentWithLineSearch(int iterations) {
        this(new GradientDescentWithLineSearchPrm(iterations));
    }
    
    public GradientDescentWithLineSearch(GradientDescentWithLineSearchPrm prm) {
        this.prm = prm;
        iterCount = 0;
    }

    /**
     * Maximize the function starting at the given initial point.
     */
    @Override
    public boolean maximize(DifferentiableFunction function, double[] point) {
        return optimize(function, point, true);
    }

    /**
     * Minimize the function starting at the given initial point.
     */
    public boolean minimize(DifferentiableFunction function, double[] point) {
        return optimize(function, point, false);
    }

    private boolean optimize(DifferentiableFunction function, double[] point, final boolean maximize) {        
        assert (function.getNumDimensions() == point.length);
        double[] gradient = new double[point.length];
        
        int passCount = 0;
        double passCountFrac = 0;
        for (iterCount=0; iterCount < prm.iterations; iterCount++) {
            function.setPoint(point);
            
            // Get the current value of the function.
            double value = function.getValue();
            log.info(String.format("Function value on batch = %g at iteration = %d", value, iterCount));
            
            // Get the gradient of the function.
            Arrays.fill(gradient, 0.0);
            function.getGradient(gradient);
            assert (gradient.length == point.length);
            
            // Take a step in the direction of the gradient.
            double lr = lineSearch(function, maximize, point, gradient);
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
        log.info(String.format("Function value on all examples = %g at iteration = %d on pass = %.2f", value, iterCount, passCountFrac));
        
        // We don't test for convergence.
        return false;
    }

    // Armijo rule: ensures that the step size decreases f 'sufficiently'
    public static boolean sufficientDecrease(double new_value, double value, double step, double c1, double [] d, double [] g) {      
        return new_value <= value+c1*step*Vectors.dotProduct(d, g);
    }

    // Curvature rule: ensures that the slope has been reduced 'sufficiently'
    public static boolean sufficientCurvature(double [] d, double [] g, double [] new_g, double c2) {
        return Vectors.dotProduct(d, new_g) >= c2*Vectors.dotProduct(d, g);
    }

    // TODO: quadratic interpolation
    private double lineSearch(DifferentiableFunction f, boolean maximize, double [] x, double [] g) {
    	
    	double step = prm.initial_step;
        int iter = 0;
        double [] d = g;
        double value = f.getValue();
        double [] new_g = new double[f.getNumDimensions()];
        double [] new_x = new double[f.getNumDimensions()];
        boolean decrease;
		boolean curvature;
		do {                        
        	for(int i=0; i<f.getNumDimensions(); i++) {
        		new_x[i] = x[i] + step*g[i];
        	}
        	f.setPoint(new_x);
        	double new_value = f.getValue();
        	f.getGradient(new_g);
        	
        	curvature = sufficientCurvature(d, g, new_g, prm.c2);
        	decrease  = sufficientDecrease(new_value, value, step, prm.c1, d, g);
        	
        	// Contract the step size
        	step *= prm.tau;
        	
        } while(!curvature && !decrease);
       
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