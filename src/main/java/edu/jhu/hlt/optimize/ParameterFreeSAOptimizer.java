package edu.jhu.hlt.optimize;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.hlt.optimize.function.ConstrainedFunction;
import edu.jhu.hlt.optimize.propose.Proposable;
import edu.jhu.hlt.util.math.Vectors;
import edu.jhu.prim.util.random.Prng;
import edu.jhu.prim.vector.IntDoubleVector;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.MatrixUtils;

/**
 * "An Efficient Simulated Annealing Schedule"
 * J Lam
 * 
 * "On the Design of an Adaptive Simulated Annealing ..."
 * VA Cicirello
 * 
 * These papers suggest a bunch of heuristics to pick the temperature schedule.
 * The generating function is an N-dimension Gaussian centered at the current point.
 * 
 * FIXME: Not tested with the new interface.
 * 
 * @author noandrews
 */
public class ParameterFreeSAOptimizer implements Optimizer<ConstrainedFunction> {

	static Logger log = LoggerFactory.getLogger(ParameterFreeSAOptimizer.class);
	
	final double eps = 1e-4;
	int max_trials = 100; // maximum # of trials to find a point inside the bounds
	double K;             // damping to track acceptance rate
	double T = 0.5;       // initial accept temperature: its adapted from here
	IntDoubleVector curr;
	int max_eval;
	IntDoubleVector covar;
	MultivariateNormalDistribution N;
	ConstrainedFunction f;
	
	public ParameterFreeSAOptimizer(ConstrainedFunction f, int max_eval) {
		this.f = f;
		this.max_eval = max_eval;
		this.K = (double)max_eval * 0.1;
	}
	
	private void updateCovar(double frac) {
		for(int i=0; i<f.getNumDimensions(); i++) {
			double L = (f.getBounds().getUpper(i)-f.getBounds().getLower(i));
			covar.set(i, (1-frac)*L);
			//log.info("covar " + i + " = " + covar[i]);
		}
	}

	private void nextRandomState(IntDoubleVector curr, IntDoubleVector next) {
		
		// FIXME: take into account the current state and the temperature
		if(f instanceof Proposable) {
			Proposable p = (Proposable)f;
			double [] pt = p.samplePoint();
			for(int i=0; i<f.getNumDimensions(); i++) {
				next.set(i, pt[i]);
			}
			return;
		}
		
		double [] Carr = new double[f.getNumDimensions()];
		for(int i=0; i<Carr.length; i++) {
			Carr[i] = covar.get(i); 
		}
		double [][] C = MatrixUtils.createRealDiagonalMatrix(Carr).getData();
		double [] m = new double[f.getNumDimensions()];
		// FIXME: need a method to get an array from a vector
		double [] pt = new double[f.getNumDimensions()];
		for(int i=0; i<pt.length; i++) {
			pt[i] = curr.get(i);
		}
		N = new MultivariateNormalDistribution(pt, C);
		double [] sample = N.sample();
		for(int i=0; i<sample.length; i++) {
			next.set(i, sample[i]);
		}
	}
	
	@Override
	public boolean minimize(ConstrainedFunction f, IntDoubleVector initial) {
		
		curr = initial.copy();
		covar = initial.copy();	
		IntDoubleVector next = initial.copy();
		IntDoubleVector best = next.copy();
		double curr_E = f.getValue(initial);
		double best_E = curr_E;
		double next_E;
		
		T = 0.5;
		double accept_rate = 0.5;
		int total_accept = 0;
		
		log.info("[SA] initial energy = " + curr_E);
		
		for(int i=1; i<max_eval; i++) {
			
			double frac = (double)i/(double)max_eval;
			
			// Set the variance of the proposal distribution
			updateCovar(frac);
			
			for(int j=0; j<max_trials; j++) {
				nextRandomState(curr, next);
				if(f.getBounds().inBounds(next)) {
					break;
				}
			}
			
			//log.info( "euc(curr,next) = " + Vectors.euclid(curr, next) );
				
			next_E = f.getValue(next);
			
			//log.info("next=("+next[0]+"), energy = " + next_E);
			
			boolean accepted = false;
			if(next_E < curr_E) {
				// accept move
				accept_rate = (1.0/500)*(499.0*accept_rate+1);
				accepted = true;
			} else {
				double r = Prng.nextDouble();
				if (r<Math.exp((curr_E-next_E)/T)) {
					// accept move
					accept_rate = (1.0/500)*(499.0*accept_rate+1);
					accepted = true;
				} else {
					// reject move
					accept_rate = (1.0/500)*(499.0*accept_rate);
				}
			}
			
			if(accepted) {
				//log.info("accepted!");
				total_accept += 1;
				
				// Update current point and associated energy
				curr_E = next_E;
				curr = next.copy();
				
				// Check if current updated point is the best so far
				if(curr_E < best_E) {
					best_E = curr_E;
					best = curr.copy();
				}
			}
			
			double lamRate = 0.44;
			if(frac < 0.15) {
				lamRate = 0.44 + 0.56*Math.pow(560.0, -frac/0.15);
			} 
			if(frac >= 0.15 && frac < 0.65) {
				lamRate = 0.44;
			}
			if(frac >= 0.65) {
				lamRate = 0.44 * Math.pow(440.0, -(frac-0.65)/0.35);
			}
			//log.info("accept rate = " + accept_rate);
			//log.info("lam rate = " + lamRate);
			
			// Update T to get desired acceptance rate
			T = (1.0 - (accept_rate-lamRate)/K)*T;
			
			//log.info("iter " + i + ": T="+T+" best_E="+best_E);
		}
				
		//log.info("[SA] total accepted = " + total_accept);
		//log.info("[SA] final point = " + best[0]);
		//log.info("[SA] final energy = " + best_E);
		
		// Set the best point
		initial = best;
		
		return true;
	}

	@Override
	public boolean maximize(ConstrainedFunction function, IntDoubleVector point) {
		throw new UnsupportedOperationException();
	}
	
}
