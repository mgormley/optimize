package edu.jhu.hlt.optimize;

import org.apache.log4j.Logger;

import edu.jhu.hlt.util.Prng;

/**
 * "An Efficient Simulated Annealing Schedule"
 * J Lam
 * 
 * "On the Design of an Adaptive Simulated Annealing ..."
 * VA Cicirello
 * 
 * These papers suggest a bunch of heuristics to pick the temperature schedule.
 * 
 * @author noandrews
 */
public class ParameterFreeSAOptimizer extends    Optimizer<ConstrainedFunction>
									  implements Minimizer<ConstrainedFunction> {

	static Logger log = Logger.getLogger(ParameterFreeSAOptimizer.class);
	
	int max_trials = 100;
	double T = 0.5;
	double [] curr;
	int max_eval;
	
	public ParameterFreeSAOptimizer(ConstrainedFunction f, int max_eval) {
		super(f);
		curr = new double[f.getNumDimensions()];
		this.max_eval = max_eval;
	}

	// This uses the VFSA proposal
	private void nextRandomState(double [] curr, double [] next) {
		for(int i=0;i<curr.length; i++) {
			double r = VFSAOptimizer.getCauchy(T);
			next[i] = curr[i] + r*(f.getBounds().getUpper(i) - f.getBounds().getLower(i));
		}
	}
	
	@Override
	public boolean minimize(ConstrainedFunction f, double[] initial) {
		
		double [] next = new double[curr.length];
		double [] best = new double[curr.length];
		double curr_E = f.getValue();
		double best_E = curr_E;
		double next_E;
		
		T = 0.5;
		double accept_rate = 0.5;
		
		for(int i=1; i<max_eval; i++) {
			
			nextRandomState(curr, next);
				
			f.setPoint(next);
			next_E = f.getValue();
			
			boolean accepted = false;
			if(next_E < curr_E) {
				// accept move
				accept_rate = (1.0/500)*(499.0*accept_rate+1);
				accepted = true;
			} else {
				double r = Prng.nextDouble();
				if (r<Math.exp(curr_E-next_E)/T) {
					// accept move
					accept_rate = (1.0/500)*(499.0*accept_rate+1);
					accepted = true;
				} else {
					// reject move
					accept_rate = (1.0/500)*(499.0*accept_rate);
				}
			}
			
			if(accepted) {
				curr_E = next_E;
				for(int j=0; j<curr.length; j++) {
					curr[j] = next[j];
				}
				
				if(curr_E > best_E) {
					best_E = curr_E;
					for(int j=0; j<curr.length; j++) {
						best[j] = curr[j];
					}
				}
			}
			
			double lamRate = 0.44;
			double frac = i/(double)max_eval;
			if(frac < 0.15) {
				lamRate = 0.44 + 0.56*Math.pow(560.0, -frac/0.15);
			} 
			if(frac >= 0.15 && frac < 0.65) {
				lamRate = 0.44;
			}
			if(frac >= 0.65) {
				lamRate = 0.44 * Math.pow(440.0, -(frac-0.65)/0.35);
			}
			if(accept_rate > lamRate) {
				T *= 0.999;
			} else {
				T /= 0.999;
			}
			
			log.info("iter " + i + ": T="+T+" best_E="+best_E);
		}
				
		// Set the best point
		f.setPoint(best);
		
		return true;
	}

	@Override
	public boolean minimize() {
		return minimize(f, curr);
	}
	
}
