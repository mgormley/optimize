package edu.jhu.hlt.optimize;


import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.junit.Test;

import junit.framework.Assert;

public class OptimizeTester {

	public List<TestFunction> getTestFunctions() {
		List<TestFunction> fs = new ArrayList<TestFunction>();
		
		fs.add(new TestFunction(new XSquared(0d)));
		
		return fs;
	}
	
	@Test
	public void testStochasticGradientDescent() {
		fail("Not yet implemented");
	}

}
