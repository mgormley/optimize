package edu.jhu.hlt.optimize;

import edu.jhu.hlt.util.Prng;

/**
 * 
 * L. Ingber, Very fast simulated re-annealing, Mathl. Comput. Modelling,Vol. 12 No. 8, pp. 967-973, 1989.
 * 
 * Notes:
 *  - This version doesn't implement re-annealing
 * 
 * Wishlist:
 *  - In some cases it might be useful to pass in a bounding box for the parameters,
 *    so that moves which go outside the bounding box can just be rejected.
 * 
 * @author noandrews
 */
public class VFSAOptimizer extends    Optimizer<Function>
                           implements Maximizer<Function>, 
                                      Minimizer<Function> { 
	
	double [] L; // characteristic length of each dimension 
				 // e.g. upper bound of dim i - lower bound of dim i
	double [] T; // temperature of each dimension
	double [] c; // control parameter of the annealing schedule
				 // larger c: faster schedule convergence
	double T0;   // initial temperature
	int eps;  
	
	public VFSAOptimizer(Function f, double [] L) {
		super(f);
		this.L = L;
	}
	
	private void nextPoint(double [] curr, double [] T, double [] next) {
		
		// Perform a random walk along each dimension
		for(int i=0; i<curr.length; i++) {
			double r = getCauchy(T[i]);
			next[i] = curr[i] + r*L[i];
		}
		
	}
	
	private double getCauchy(double T) {
		double u = Prng.nextDouble();
		return Math.signum(u-0.5d)*T*(Math.pow(1d+1d/T, Math.abs(2d*u-1d))-1d);
	}
	
	// Update
	private void schedule(double [] T, int k) {
		for(int i=0; i<T.length; i++) {
			T[i] = T0*Math.exp(-c[i]*Math.pow(k, 1d/c.length));
		}
	}
	
	@Override
	public boolean minimize(Function function, double[] initial) {
		
		this.f = function;
		f.setPoint(initial);
		
		return minimize();
	}

	private boolean optimize(boolean minimize) {
		
		double curr_E = f.getValue();
		double next_E;
		
		// Optimum so far
		double [] next_point = new double[f.getNumDimensions()];
		double [] curr_point = f.getPoint();
		double [] best_point = f.getPoint();
		double best_E = curr_E;
		
		// Stopping criteria: 
		//  - Optimum hasn't changed in the last epsilon iterations
		//  - Temperature becomes sufficiently close to 0
		int cntr = 0;
		while(true) {
			
			// Jump
			nextPoint(curr_point, T, next_point);
			
			// Accept/reject
			boolean accept = 1d/(1d+Math.exp(1d)) > 1; // FIXME
			
			// Update schedule
			// TODO
			
			// Check for convergence
			if(curr_E > best_E) {
				cntr = 0;
			} else {
				cntr ++;
			}
			
			if(cntr > eps) break;
		}
		
		return true;
	}
	
	@Override
	public boolean minimize() {
		
		
		
		return true;
	}

	@Override
	public boolean maximize(Function function, double[] point) {
		
		this.f = function;
		f.setPoint(point);
		
		return maximize();
	}

	@Override
	public boolean maximize() {
		// TODO Auto-generated method stub
		return false;
	}

	
	
}
