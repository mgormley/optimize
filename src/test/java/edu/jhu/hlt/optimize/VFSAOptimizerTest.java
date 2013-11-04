package edu.jhu.hlt.optimize;

import static org.junit.Assert.assertEquals;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.junit.Test;

import edu.jhu.hlt.optimize.function.Bounds;
import edu.jhu.hlt.optimize.function.ConstrainedDifferentiableFunction;
import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.FunctionOpts;
import edu.jhu.hlt.optimize.functions.Rastrigins;
import edu.jhu.hlt.util.Prng;

public class VFSAOptimizerTest {

	static Logger log = Logger.getLogger(VFSAOptimizerTest.class);
	
	@Test
	public void rastriginsTest() {
		
		BasicConfigurator.configure();
    	Logger.getRootLogger().setLevel(Level.DEBUG);
		
		Prng.seed(42);
		
		int D;
		int maxiter;
		
		// Low dimensions
		D = 2;
		maxiter = 1000;
		
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
		VFSAOptimizer opt = new VFSAOptimizer(g, 10);
		
		opt.minimize();
		
		double [] opt_point = f.getPoint();
		double opt_val = f.getValue();
		
		log.info("found opt val = " + opt_val);
	    assertEquals(0, opt_val, 1e-2);
	    
	    // see how close we are to the opt point
	    for(int i=0; i<D; i++) {
	    	assertEquals(0, opt_point[i], 1e-2);
	    }
		
		// Medium dimensions
		
	    
		// High dimensions
		
		
	}
	
}
