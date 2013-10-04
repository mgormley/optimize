package edu.jhu.hlt.optimize;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.junit.Test;

public class SGDQNCorrectedTest extends OptimizeTester {

	static Logger log = Logger.getLogger(SGDQNCorrectedTest.class);
	
	@Test
	public void simpleTest() {
		
    	BasicConfigurator.configure();
    	Logger.getRootLogger().setLevel(Level.DEBUG);
		
		for(TestFunction test : this.getTestFunctions()) {
			
			DifferentiableFunction f = (DifferentiableFunction) test.f;
			SGDQNCorrected opt       = new SGDQNCorrected(f);
			
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
