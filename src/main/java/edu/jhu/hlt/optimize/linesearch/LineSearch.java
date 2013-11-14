package edu.jhu.hlt.optimize.linesearch;

import java.util.ArrayList;
import java.util.List;

import edu.jhu.hlt.optimize.function.Bounds;
import edu.jhu.hlt.optimize.function.ConstrainedDifferentiableFunction;
import edu.jhu.hlt.optimize.function.Function;
import edu.jhu.prim.util.Lambda.FnIntDoubleToDouble;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * Basic implementation of a line search. If the initial step size is set incorrectly,
 * this algorithm will break. See ParanoidLineSearch for a more robust (slow) implementation.
 * 
 * @author noandrews
 */
public class LineSearch {
	
	Function f;
	boolean maximize;
	boolean checkCurvature   = false;
	boolean checkImprovement = true;
	boolean checkBounds      = false;
	Bounds b;
	List<Double> steps = new ArrayList<Double>(); // list of previously set step sizes
	LineSearchPrm prm;
	
	public static class LineSearchPrm {
		public int max_linesearch_iter = 100;
		public double initial_step = 0.5;
		public double tau = 0.5;
		public double c1 = 0.5;
		public double c2 = 1e-4;
		public double min_step_size = 1e-8;
	};
	
	public LineSearch(Function f, boolean maximize) {
		this.f = f;
		if(f instanceof ConstrainedDifferentiableFunction) {
			b = ((ConstrainedDifferentiableFunction)f).getBounds();
			checkBounds = true;
		}
		this.maximize = maximize;
		prm = new LineSearchPrm();
	}
	
	public LineSearch(Function f, boolean maximize, boolean checkCurvature, boolean checkImprovement) {
		this(f, maximize);
		this.checkCurvature = checkCurvature;
		this.checkImprovement = checkImprovement;
	}
	
	/**
	 * Find a step size which "sufficiently" increases the given function.
	 * 
	 * @param f				Function to optimize
	 * @param maximize		If the function is being maximized or minimized
	 * @param x				The input point
	 * @param g				Gradient of the function evaluated at x
	 * @return				Step size
	 */
	public double search(IntDoubleVector x, IntDoubleVector g) {
    	
    	double step = prm.initial_step;
    	
        int iter = 0;
        IntDoubleVector d = g.copy();
        double value = f.getValue(x);
        boolean decrease;
		boolean constraints;
				
		do {
			final double lr = step;
			
			// Scale the gradient by the learning rate.
            g.apply(new FnIntDoubleToDouble() {
                @Override
                public double call(int index, double value) {
                    if (maximize) {
                        return lr * value;
                    } else {
                        return - lr * value;
                    }
                }
            });
			
            IntDoubleVector new_x = x.copy();
            new_x.add(g);   	
        	double new_value = f.getValue(new_x);
        	
        	decrease = sufficientDecrease(new_value, value, step, prm.c1, d, g, maximize);
        	
        	if(checkBounds) {
        		constraints = satisfiesBounds(new_x);
        	} else {
        		constraints = true;
        	}
        	        	
        	if(decrease && constraints) {
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
        } while(!decrease || !constraints); 
 
        return step;
    }
	
	// Armijo rule: ensures that the step size improves f 'sufficiently'
    public boolean sufficientDecrease(double new_value, double value, double step, double c1, IntDoubleVector d, IntDoubleVector g, boolean maximize) {      
    	if(maximize) {
    		if(new_value < value) return false;
        	return new_value >= value+c1*step*d.dot(g);
        } else {
        	if(new_value > value) return false;
        	return new_value <= value+c1*step*d.dot(g);
        }
    }
    
    public boolean satisfiesBounds(IntDoubleVector new_pt) {
    	for(int i=0; i<f.getNumDimensions(); i++) {
    		if(new_pt.get(i) > b.getUpper(i) ||
    		   new_pt.get(i) < b.getLower(i)) {
    			return false;
    		}
    	}
    	return true;
    }
}
