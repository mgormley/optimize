package edu.jhu.hlt.optimize;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.junit.Test;

import edu.jhu.hlt.optimize.functions.TestFunction;


public class NaturalNewtonTest extends OptimizeTester {
	
	@Test
	public void simpleTest() {
		
    	BasicConfigurator.configure();
    	Logger.getRootLogger().setLevel(Level.DEBUG);
		
		for(TestFunction test : this.getTestFunctions()) {
			
			DifferentiableFunction f = (DifferentiableFunction) test.f;
			NaturalNewton opt        = new NaturalNewton(f);
			
			if(test.maximize) {
				opt.maximize();
			} else {
				opt.minimize();
			}
			
			test.checkValue(f.getValue());
			test.checkParam(f.getPoint());
			
		}
		
	}
	
}
