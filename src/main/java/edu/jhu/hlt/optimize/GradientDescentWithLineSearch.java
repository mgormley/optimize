package edu.jhu.hlt.optimize;

import java.util.Arrays;

import org.apache.log4j.Logger;

import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.FunctionOpts;
import edu.jhu.hlt.optimize.linesearch.ParanoidLineSearch;
import edu.jhu.hlt.util.math.Vectors;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * Gradient descent. A line search is used at each iteration to pick the step size.
 * 
 * @author noandrews
 */
public class GradientDescentWithLineSearch implements Optimizer<DifferentiableFunction> {

    /** Options for this optimizer. */
    public static class GradientDescentWithLineSearchPrm {
        /** The number of iterations to perform. */
        public int iterations = 10;
        public double min_step = 1e-8;
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
    }

    private boolean optimize(DifferentiableFunction function, IntDoubleVector point, final boolean maximize) {        
        IntDoubleVector gradient;
        ParanoidLineSearch line = new ParanoidLineSearch(function);
        
        for (iterCount=0; iterCount < prm.iterations; iterCount++) {
            
            // Get the current value of the function.
            double value = function.getValue(point);
            log.info(String.format("Function value = %g at iteration = %d", value, iterCount));
            
            
            // Get the gradient of the function.
            gradient = function.getGradient(point);
            
            // Take a step in the direction of the gradient.
            double lr = line.search(point, gradient);
            if (maximize) {
                gradient.scale(lr);
            } else {
                gradient.scale(-lr);
            }
            log.info("step size = " + lr);
            point.add(gradient);
            log.info("function value = " + function.getValue(point));
            if(lr < prm.min_step || line.converged()) {
            	log.info("converged");
            	break;
            }
        }
        
        // Get the final value of the function on all the examples.
        double value = function.getValue(point);
        log.info(String.format("Final function value = %g", value));
        
        return true;
    }
    
	@Override
	public boolean minimize(DifferentiableFunction function,
			IntDoubleVector point) {
	    return optimize(function, point, false);
	}

	@Override
	public boolean maximize(DifferentiableFunction function,
			IntDoubleVector point) {
	    return optimize(function, point, true);
	}
    
}