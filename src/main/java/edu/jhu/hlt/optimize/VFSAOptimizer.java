package edu.jhu.hlt.optimize;

import edu.jhu.hlt.util.Prng;

/**
 * L. Ingber, Very fast simulated re-annealing, Mathl. Comput. Modelling,Vol. 12 No. 8, pp. 967-973, 1989.
 * 
 * Wishlist:
 *  - In some cases it might be useful to pass in a bounding box for the parameters,
 *    so that moves which go outside the bounding box can just be rejected. I guess
 *    we need proper interfaces for constrained optimization vs unconstrained optimization.
 * 
 * @author noandrews
 */
public class VFSAOptimizer extends    Optimizer<Function>
                           implements Maximizer<Function>,
                                      Minimizer<Function> {
	
	// Acceptance-rate schedule parameters (to accept/reject proposals)
	double a_T;     // acceptance rate parameter
	double a_T0;    // initial temperature
	double a_c;     // parameter controlling schedule rate of change
	
	// Variable-specific schedule parameters (to generate proposals)
	double [] L;  // characteristic length of each dimension
				  // e.g. upper bound of dim i - lower bound of dim i
	double [] T;  // temperature of each dimension
	
	double [] c;  // control parameter of the annealing schedule
				  // larger c: faster schedule convergence
	
	double [] T0; // initial temperature
	
	// Convergence parameters
	int eps;
	double min_T = 1e-3;
	
	public VFSAOptimizer(Function f, double [] L) {
		super(f);
		this.L = L;
		
		// Initialize the temperatures to 1 initially
		a_T0 = 1d;
		T0 = new double[f.getNumDimensions()];
		for(int i=0; i<T0.length; i++) {
			T0[i] = 1d;
		}
		
		// Initialize control parameters
		a_c = 1d;
		c = new double[f.getNumDimensions()];
		for(int i=0; i<c.length; i++) {
			c[i] = 1d;
		}
	}
	
	// TODO: I think this needs specific versions for min and max
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
	private void updateSchedules(int k) {
		
		// Acceptance rate schedule
		a_T = a_T0*Math.exp(-a_c*Math.pow(k, 1d/T.length));
		
		// Dimension-specific schedules
		for(int i=0; i<T.length; i++) {
			T[i] = T0[i]*Math.exp(-c[i]*Math.pow(k, 1d/T.length));
		}
		
	}
	
	@Override
	public boolean minimize(Function function, double[] initial) {
		
		this.f = function;
		f.setPoint(initial);
		
		return minimize();
	}

	private boolean accept(double delta, boolean minimize) {
		if(minimize) {
			if(delta < 0) return true;
			double Tprob = Math.exp(-delta/a_T);
			if(Tprob > Prng.nextDouble()) return true;
		} else {
			if(delta > 0) return true;
			double Tprob = Math.exp(-delta/a_T);
			if(Tprob > Prng.nextDouble()) return true;
		}
		return false;
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
		int k = 0;
		while(true) {
			
			// Set the temperature
			updateSchedules(k);
			
			// Jump
			nextPoint(curr_point, T, next_point);
			
			// Update objective function
			next_E = f.getValue(curr_point);
			
			// Compute objective function delta
			double delta = next_E - curr_E;
			
			// Accept/reject
			if(accept(delta, minimize)) {
				curr_point = next_point;
				curr_E = next_E;
				
				// Update optimum so far
				if(next_E > best_E) {
					best_point = curr_point;
					best_E = next_E;
				}
			}
			
			// Check for convergence
			if(curr_E > best_E) {
				cntr = 0;
			} else {
				cntr ++;
			}
			if(cntr > eps) break; // No change in optimum
			if(a_T<min_T)  break; // Negligible temperature
			
			// Update iteration number
			k += 1;
		}
		
		// Set the function to its optimum
		f.setPoint(best_point);
		
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
