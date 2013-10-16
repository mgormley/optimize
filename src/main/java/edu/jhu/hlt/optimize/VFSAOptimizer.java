package edu.jhu.hlt.optimize;

import org.apache.log4j.Logger;

import edu.jhu.hlt.util.Prng;
import edu.jhu.hlt.util.math.Vectors;

/**
 * SA variant:
 *   L. Ingber, Very fast simulated re-annealing 
 *   Mathl. Comput. Modelling,Vol. 12 No. 8, pp. 967-973, 1989.
 * 
 * Heuristic to pick a starting temperature for the acceptance function:
 *   Ben-Ameur, Walid. Computing the initial temperature of simulated annealing. 
 *   Comp. Opt. and Appl, Vol. 29 No. 3 pp 369-385, 2004.
 * 
 * @author noandrews
 */
public class VFSAOptimizer extends    Optimizer<DifferentiableFunction>
                           implements Maximizer<DifferentiableFunction>,
                                      Minimizer<DifferentiableFunction> {
	
	static Logger log = Logger.getLogger(VFSAOptimizerTest.class);
	
	// Acceptance-rate schedule parameters (to accept/reject proposals)
	double a_T;     // acceptance rate parameter
	double a_T0;    // initial temperature (set adaptively based on desired acceptance rate)
	double a_c;     // parameter controlling schedule rate of change
	int a_k;        // number of accepted points
	
	// Variable-specific schedule parameters (to generate proposals)
	double [] L;  // characteristic length of each dimension
				  // e.g. upper bound of dim i - lower bound of dim i
	double [] T;  // temperature of each dimension	
	double [] c;  // control parameter of the annealing schedule
				  // larger c: faster schedule convergence
	int [] k;     // iteration counters (these are used when "re-annealing")
	double [] T0; // initial temperatures
	
	// Convergence parameters
	final int eps = 100;
	final double min_T = 1e-3;
	
	// Other magic numbers
	int samples_per_temp;            // (see constructor)
	int temps_between_reanneal = 10; // number of cooling iterations between adaptation steps
	double desired_accept = 2d/3d;   // desired acceptance rate
	double a_T1 = 1d;                // T0 for trial SA run used in setting actual T0 adaptively for desired_accept
	
	public VFSAOptimizer(DifferentiableFunction f, double [] L) {
		super(f);
		this.L = L;
		
		samples_per_temp = (int) (0.1 * f.getNumDimensions());
		
		// Initialize the temperatures to 1 initially
		a_T0 = estimateStartingTemp();
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
	
	// Estimates starting temperature according to desired acceptance rate
	private double estimateStartingTemp() {
		// TODO
		return 1d;
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
	
	// Update T for the acceptance rate and for each input variable
	private void updateSchedules() {
		
		// Acceptance rate schedule
		a_T = a_T0*Math.exp(-a_c*Math.pow(a_k, 1d/T.length));
		
		// Dimension-specific schedules
		for(int i=0; i<T.length; i++) {
			T[i] = T0[i]*Math.exp(-c[i]*Math.pow(k[i], 1d/T.length));
		}
		
	}
	
	// Re-annealing
	private void reAnneal(double [] best_point) {
	
		// Step 1: Compute sensitivities
		double [] s = new double[f.getNumDimensions()];
		double [] grad = new double[f.getNumDimensions()];
		f.setPoint(best_point);
		f.getGradient(grad);
		for(int i=0; i<s.length; i++) {
			s[i] = grad[i];
		}
		
		// Step 2: Compute updated temperatures
		double s_max = Vectors.max(s);
		double [] Tbar = new double[f.getNumDimensions()];
		for(int i=0; i<s.length; i++) {
			Tbar[i] = T[i] * (s_max/s[i]) ;
		}
		
		// Step 3: Update the iteration counters for the desired temperatures
		for(int i=0; i<s.length; i++) {
			k[i] = (int) Math.pow(Math.log(T0[i]/Tbar[i])/c[i], f.getNumDimensions());
		}
		
	}
	
	@Override
	public boolean minimize(DifferentiableFunction function, double[] initial) {
		
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
		a_k = 0;
		for(int i=0; i<f.getNumDimensions(); i++) {
			k[i] = 0;
		}
		int iter = 0;
		while(true) {
			
			// Set the temperatures
			updateSchedules();
			
			for(int m=0; m<samples_per_temp; m++) {
				// Jump
				nextPoint(curr_point, T, next_point);
			
				// Update objective function
				next_E = f.getValue(curr_point);
			
				// Compute objective function delta
				double delta = next_E - curr_E;
			
				// Accept/reject
				if(accept(delta, minimize)) {
					a_k += 1;
					
					curr_point = next_point;
					curr_E = next_E;
				
					// Update optimum so far
					if(next_E > best_E) {
						best_point = curr_point;
						best_E = next_E;
						cntr = 0;
					}
				}
			}
		
			cntr ++;
			
			// Check for convergence
			if(cntr > eps) break; // No change in optimum
			if(a_T<min_T)  break; // Negligible temperature
			
			if(++iter % temps_between_reanneal == 0) {
				reAnneal(best_point);
			}
			
			// Increment k's for the T schedules
			for(int i=0; i<f.getNumDimensions(); i++) {
				k[i] += 1;
			}
		}
		
		// Set the function to its optimum
		f.setPoint(best_point);
		
		return true;
	}
	
	@Override
	public boolean minimize() {
		return optimize(true);
	}

	@Override
	public boolean maximize(DifferentiableFunction function, double[] point) {
		
		this.f = function;
		f.setPoint(point);
		
		return maximize();
	}

	@Override
	public boolean maximize() {
		return optimize(false);
	}

	
	
}
