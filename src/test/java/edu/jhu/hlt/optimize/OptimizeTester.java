package edu.jhu.hlt.optimize;


import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

public class OptimizeTester {

	public List<TestFunction> getTestFunctions() {
		List<TestFunction> fs = new ArrayList<TestFunction>();
		
		fs.add(new TestFunction(new XSquared(+1d), 0d, 1e-4, new double[] {0d}, 1e-4));
		fs.add(new TestFunction(new XSquared(-1d), 0d, 1e-4, new double[] {0d}, 1e-4));
		
		return fs;
	}
	
	@Test
	public void testStochasticGradientDescent() {
		List<TestFunction> fs = getTestFunctions();
		for(TestFunction f : fs) {
			StochasticGradientDescent sgd = new StochasticGradientDescent((DifferentiableRealScalarFunction) f.getFunction(), 0d, 0d, 0d);
			f.checkValue(sgd.val());
			f.checkParam(sgd.getFunction().get());
		}
	}

}
