package edu.jhu.hlt.optimize;

import java.util.ArrayList;
import java.util.List;

import edu.jhu.hlt.util.math.Vectors;

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
	public double search(double [] x, double [] g) {
    	
    	double step = prm.initial_step;
    	
        int iter = 0;
        double [] d = g;
        double value = f.getValue();
        double [] new_x = new double[f.getNumDimensions()];
        boolean decrease;
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
    public boolean sufficientDecrease(double new_value, double value, double step, double c1, double [] d, double [] g, boolean maximize) {      
    	if(maximize) {
    		if(new_value < value) return false;
        	return new_value >= value+c1*step*Vectors.dotProduct(d, g);
        } else {
        	if(new_value > value) return false;
        	return new_value <= value+c1*step*Vectors.dotProduct(d, g);
        }
    }
    
    public boolean satisfiesBounds(double [] new_pt) {
    	for(int i=0; i<f.getNumDimensions(); i++) {
    		if(new_pt[i] > b.getUpper(i) ||
    		   new_pt[i] < b.getLower(i)) {
    			return false;
    		}
    	}
    	return true;
    }
}
