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

	protected static Function negate(Function f) {
		return new FunctionOpts.NegateFunction((DifferentiableFunction) f);
	}
	
	protected List<TestFunction> getTestFunctions() {
		List<TestFunction> fs = new ArrayList<TestFunction>();
		
		// FIXME: These bounds are probably too forgiving
		fs.add(new TestFunction(new XSquared(+1d), false, 0d, 1e-2, new double[] {0d}, 1e-2));
		fs.add(new TestFunction(new XSquared(-1d), false, 0d, 1e-2, new double[] {0d}, 1e-2));
		fs.add(new TestFunction(negate(new XSquared(+1d)), false, 0d, 1e-2, new double[] {0d}, 1e-2));
		fs.add(new TestFunction(negate(new XSquared(-1d)), false, 0d, 1e-2, new double[] {0d}, 1e-2));
				
		return fs;
	}

}
