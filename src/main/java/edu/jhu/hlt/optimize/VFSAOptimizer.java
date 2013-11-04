package edu.jhu.hlt.optimize;

import org.apache.log4j.Logger;

import edu.jhu.hlt.optimize.function.ConstrainedDifferentiableFunction;
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
public class VFSAOptimizer extends    Optimizer<ConstrainedDifferentiableFunction>
                           implements Maximizer<ConstrainedDifferentiableFunction>,
                                      Minimizer<ConstrainedDifferentiableFunction> {
	
	static Logger log = Logger.getLogger(VFSAOptimizer.class);
	
	// Acceptance-rate schedule parameters (to accept/reject proposals)
	double a_T;     // acceptance rate parameter
	double a_T0;    // initial temperature (set adaptively based on desired acceptance rate)
	double a_c;     // parameter controlling schedule rate of change
	int a_k;        // number of accepted points
	
	// Variable-specific schedule parameters (to generate proposals)
	double [] A;  // minimum value of each dimension
	double [] B;  // maximum value of each dimension
	double [] L;  // characteristic length of each dimension
				  // e.g. upper bound of dim i - lower bound of dim i
	double [] T;  // temperature of each dimension	
	double [] c;  // control parameter of the annealing schedule
				  // larger c: faster schedule convergence
	int [] k;     // iteration counters (these are used when "re-annealing")
	double [] T0; // initial temperatures
	
	// Convergence parameters
	int max_iter = 1000;
	final int eps = 1000;
	final double min_T = 1e-3;
	
	// Other magic numbers
	final double MAX_TEMP = 1e5;
	final double START_TEMP = 1e-2;  // heuristics will increase this to find a starting temperature
	final int MAX_TRIALS = 100;      
	int samples_per_temp;            // (see constructor)
	int temps_between_reanneal = 10; // number of cooling iterations between adaptation steps
	double desired_accept = 0.90;    // desired initial acceptance rate
	
	// Options
	boolean reanneal = false;
	
	// Diagnostics
	int naccept;
	int nsamples;
	
	public VFSAOptimizer(ConstrainedDifferentiableFunction f, int maxiter) {
		super(f);
		
		this.max_iter = maxiter;
		
		this.L = new double[f.getNumDimensions()];
		log.info("dim(f) = " + f.getNumDimensions());
		for(int i=0; i<f.getNumDimensions(); i++) {
			this.L[i] = f.getBounds().getUpper(i)-f.getBounds().getLower(i);
			log.info("L["+i+"]="+L[i]);
		}
		
		samples_per_temp = (int) (2.0 * f.getNumDimensions());
		log.info("Samples at each temperature: " + samples_per_temp);
		
		// Initialize the temperatures to 1 initially
		a_T0 = 100d;
		T0 = new double[f.getNumDimensions()];
		T = new double[f.getNumDimensions()];
		for(int i=0; i<T0.length; i++) {
			T0[i] = 100d;
		}
		
		// Initialize parameters of the annealing schedules
		a_c = 1.0;
		c = new double[f.getNumDimensions()];
		for(int i=0; i<c.length; i++) {
			c[i] = 1.0;
		}
		
		// Counters
		a_k = 0;
		k = new int[f.getNumDimensions()];
		for(int i=0; i<f.getNumDimensions(); i++) {
			k[i] = 0;
		}
	}
	
	public void setAcceptanceT0(double t) {
		this.a_T0 = t;
	}
	
	public void setAcceptanceC(double c) {
		this.a_c = c;
	}
	
	public void setInitialT(double t) {
		for(int i=0; i<f.getNumDimensions(); i++) {
			this.T0[i] = t;
		}
	}
	
	public void setScheduleC(double c) {
		for(int i=0; i<f.getNumDimensions(); i++) {
			this.c[i] = c;
		}
	}
	
	// Estimates starting temperature according to desired acceptance rate
	private double estimateStartingTemp() {
		// TODO using the Ben-Ameur method
		return 1d;
	}
	
	// Takes the currently set point and tries to sample around it
	private void heuristicStartingTemp(boolean minimize) {
		// Set the initial temperature really high
		this.setAcceptanceT0(START_TEMP);
		this.setInitialT(START_TEMP);
		double naccept;
		double rate;
		int iter = 0;
		int trials = 1000;
		double curr_T0 = START_TEMP;
		double [] curr = f.getPoint();
		double [] start = new double[f.getNumDimensions()];
		for(int i=0; i<f.getNumDimensions(); i++) {
			start[i] = curr[i];
		}
		double curr_E = f.getValue();
		log.info("start E = " + curr_E);
		double [] next = new double[f.getNumDimensions()];
		do {
			curr_T0 *= 2.0;
			
			this.setAcceptanceT0(curr_T0);
			this.setInitialT(curr_T0);
			this.updateSchedules();
			
			naccept = 0;
			for(int i=0; i<trials; i++) {
				nextPoint(start, next);
				
				if(!f.getBounds().inBounds(next)) {
					continue;
				}
				
				double next_E = f.getValue(next);
				//log.info("next E = " + next_E);
				if(accept(next_E-curr_E,minimize)) {
					naccept+=1;
				}
			}
			rate = naccept/trials;
			if(++iter > this.MAX_TRIALS) break;
			if(curr_T0 > this.MAX_TEMP) break;
			log.info("accept rate @ " + curr_T0 + " = " + rate);
		} while(rate < desired_accept);
		
		log.info("starting T0 = " + curr_T0);
				
		// Set T0
		this.setAcceptanceT0(curr_T0);
		this.setInitialT(curr_T0);
	}
	
	private void nextPoint(double [] curr, double [] next) {
		
		// Perform a random walk along each dimension
		for(int i=0; i<curr.length; i++) {
			double r = getCauchy(T[i]);
			next[i] = curr[i] + r*L[i];
		}
		
	}
	
	public static double getCauchy(double t) {
		double u = Prng.nextDouble();
		double x = Math.pow(1.0+1.0/t, Math.abs(2.0*u-1));
		return Math.signum(u-0.5d)*t*(x-1.0);
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
			Tbar[i] = T[i] * (s_max/s[i]);
		}
		
		// Step 3: Update the iteration counters for the desired temperatures
		for(int i=0; i<s.length; i++) {
			k[i] = (int) Math.pow(Math.log(T0[i]/Tbar[i])/c[i], f.getNumDimensions());
		}
		
	}
	
	@Override
	public boolean minimize(ConstrainedDifferentiableFunction function, double[] initial) {
		
		this.f = function;
		f.setPoint(initial);
		
		return minimize();
	}

	private boolean accept(double delta, boolean minimize) {
		if(minimize) {
			if(delta < 0) return true;
			double Tprob = Math.exp(-delta/a_T);
			//log.info("Accept prob = " + Tprob);
			if(Tprob > Prng.nextDouble()) return true;
		} else {
			if(delta > 0) return true;
			double Tprob = Math.exp(-delta/a_T);
			if(Tprob > Prng.nextDouble()) return true;
		}
		return false;
	}
	
	private boolean optimize(boolean minimize) {
		
		// Guess some hyperparameters
		this.heuristicStartingTemp(minimize);
		
		double curr_E = f.getValue();
		double next_E;
		
		log.info("initial energy = " + curr_E);
		
		// Optimum so far
		double [] next_point = new double[f.getNumDimensions()];
		double [] curr_point = new double[f.getNumDimensions()];
		double [] best_point = new double[f.getNumDimensions()];
		
		// Initialize current and best points to starting point
		for(int i=0; i<f.getNumDimensions(); i++) {
			curr_point[i] = f.getPoint()[i];
			best_point[i] = curr_point[i];
		}
		
		double best_E = curr_E;
		
		// Stopping criteria: 
		//  - Optimum hasn't changed in the last epsilon iterations
		//  - Temperature becomes sufficiently close to 0
		int cntr = 0;
		int iter = 0;
		int naccept_per_temp;
		while(true) {
			
			// Set the temperatures
			updateSchedules();
			
			naccept_per_temp = 0;
			
			// Try to make samples_per_temp moves
			for(int m=0; m<samples_per_temp; m++) {
				
				// Jump
				nextPoint(curr_point, next_point);
				nsamples += 1;
			
				// Update objective function
				next_E = f.getValue(next_point);
				
				log.info("next E = " + next_E + " (best="+best_E+")");
			
				// Compute objective function delta
				double delta = next_E - curr_E;
				log.info("delta = " + delta);
			
				boolean in_bounds = f.getBounds().inBounds(next_point);
				
				// Accept/reject
				if(in_bounds && accept(delta, minimize)) {
					
					log.info("\t======= accepted! =======\t");
					
					a_k += 1;
					
					for(int i=0; i<f.getNumDimensions(); i++) {
						curr_point[i] = next_point[i];
					}
					
					curr_E = next_E;
				
					naccept_per_temp += 1;
					naccept += 1;
					
					// Update optimum so far
					if(minimize) {
						if(curr_E < best_E) {
							best_point = curr_point;
							best_E = curr_E;
							cntr = 0;
						} else {
							cntr ++;
						}
					} 
					else {
						if(curr_E > best_E) {
							best_point = curr_point;
							best_E = curr_E;
							cntr = 0;
						} else {
							cntr ++;
						}
					}
				}
			}
			
			double accept_rate = (double)naccept_per_temp/(double)samples_per_temp*100.0;
			log.info("Accept rate at T="+a_T+": " + accept_rate);
			
			// Check for convergence
			if(cntr > eps) {
				log.info("CONVERGED: no change in optimum");
				break; // No change in optimum
			}
			if(a_T<min_T) {
				log.info("CONVERGED: negligible temperature");
				break; // Negligible temperature
			}
			if(++iter % temps_between_reanneal == 0) {
				if(reanneal) {
					reAnneal(best_point);
				}
			}
			if(iter>this.max_iter) {
				log.info("CONVERGED: max iterations reached");
				break;
			}
			
			// Increment k's for the T schedules
			for(int i=0; i<f.getNumDimensions(); i++) {
				k[i] += 1;
			}
			
			// XX Debug XX
			for(int i=0; i<f.getNumDimensions(); i++) {
				log.info("T["+i+"]="+T[i]);
			}
			log.info("Acceptance temperature: " + a_T);
			log.info("Acceptance rate: " + (1.0*naccept/nsamples));
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
	public boolean maximize(ConstrainedDifferentiableFunction function, double[] point) {
		
		this.f = function;
		f.setPoint(point);
		
		return maximize();
	}

	@Override
	public boolean maximize() {
		return optimize(false);
	}

	
	
}
