package edu.jhu.hlt.optimize;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.junit.Test;

import edu.jhu.hlt.util.Prng;

public class VFSAOptimizerTest {

	static Logger log = Logger.getLogger(VFSAOptimizerTest.class);
	
	@Test
	public void rastriginsTest() {
		
		BasicConfigurator.configure();
    	Logger.getRootLogger().setLevel(Level.DEBUG);
		
		Prng.seed(42);
		
		int D;
		
		// Low dimensions
		D = 2;
		DifferentiableFunction f = new Rastrigins(D);
		// The rastrigin optimum is at vec(0)
		
		// Optimization bounds: −5.12 ≤ xi ≤ 5.12
		double [] L = new double[D];
		double [] U = new double[D];
		double [] start = new double[D];
		for(int i=0; i<D; i++) {
			L[i] = -5.12;
			U[i] = 5.12;
			start[i] = Prng.nextDouble();
		}
		f.setPoint(start);
		Bounds b = new Bounds(L, U);
		ConstrainedDifferentiableFunction g = new FunctionOpts.DifferentiableFunctionWithConstraints(f, b);
		
		VFSAOptimizer opt = new VFSAOptimizer(g);
		opt.minimize();
		
		double [] opt_point = f.getPoint();
		double opt_val = f.getValue();
		
		log.info("found opt val = " + opt_val);
		
		// Medium dimensions
		
		// High dimensions
		
		
	}
	
}
