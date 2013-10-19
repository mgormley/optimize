package edu.jhu.hlt.optimize;

import static org.junit.Assert.assertEquals;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.junit.Test;

import edu.jhu.hlt.util.JUnitUtils;
import edu.jhu.hlt.util.math.Vectors;

public class ConstrainedGradientDescentWithLineSearchTest {

	static Logger log = Logger.getLogger(ConstrainedGradientDescentWithLineSearchTest.class);
	
	   @Test
	    public void testNegXSquared() {
		   BasicConfigurator.configure();
		   Logger.getRootLogger().setLevel(Level.DEBUG);
		   
	       ConstrainedGradientDescentWithLineSearch opt = new ConstrainedGradientDescentWithLineSearch(25);
	       double[] max = new double[]{8.5}; // initial pt at x=2
	       double[] lower = new double[]{1.0};
	       double[] upper = new double[]{100.0};
	       Bounds b = new Bounds(lower, upper);
	       DifferentiableFunction f = new FunctionOpts.NegateFunction(new XSquared());
	       ConstrainedDifferentiableFunction g = new FunctionOpts.DifferentiableFunctionWithConstraints(f, b);
	       opt.maximize(g, max);
	       log.info("found opt = " + max[0]);
	       assertEquals(1.0, max[0], 1e-3);
	    }
	    
	    @Test
	    public void testXSquared() {
	    	BasicConfigurator.configure();
			Logger.getRootLogger().setLevel(Level.DEBUG);
			   
		    ConstrainedGradientDescentWithLineSearch opt = new ConstrainedGradientDescentWithLineSearch(25);
		    double[] min = new double[]{8.5}; // initial pt at x=2
		    double[] lower = new double[]{1.0};
		    double[] upper = new double[]{100.0};
		    Bounds b = new Bounds(lower, upper);
		    DifferentiableFunction f = new XSquared();
		    ConstrainedDifferentiableFunction g = new FunctionOpts.DifferentiableFunctionWithConstraints(f, b);
		    opt.minimize(g, min);
		    log.info("found opt = " + min[0]);
		    assertEquals(1.0, min[0], 1e-3);
	    }
}