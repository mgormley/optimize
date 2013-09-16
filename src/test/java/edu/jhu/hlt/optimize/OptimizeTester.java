package edu.jhu.hlt.optimize;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

/**
 * Basic test class -- should be replaced with something more flexible.
 * 
 * @author Nicholas Andrews
 */
public class OptimizeTester {

	public List<TestFunction> getTestFunctions() {
		List<TestFunction> fs = new ArrayList<TestFunction>();
		
		fs.add(new TestFunction(new XSquared(+1d), 0d, 1e-4, new double[] {0d}, 1e-3));
		fs.add(new TestFunction(new XSquared(-1d), 0d, 1e-4, new double[] {0d}, 1e-3));
		
		return fs;
	}
	
	@Test
	public void testStochasticGradientDescent() {
		List<TestFunction> fs = getTestFunctions();
		for(TestFunction f : fs) {
			StochasticGradientDescent sgd = new StochasticGradientDescent((DifferentiableRealScalarFunction) f.getFunction(), 1);
			f.checkValue(sgd.val());
			f.checkParam(sgd.getFunction().get());
		}
	}

}
