package edu.jhu.hlt.util.math;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Test;

import edu.jhu.hlt.optimize.Function;
import edu.jhu.hlt.optimize.XSquared;

public class GPRegressionTest {

	@Test
	public void regressionTest() {
		
    	Kernel kernel = new SquaredExpKernel();
    	Function f = new XSquared(0);
    	
    	// 10 training (x,y) pairs
    	RealMatrix X = MatrixUtils.createRealMatrix(1, 10);
		RealVector y = new ArrayRealVector(10);
		for(int i=0; i<10; i++) {
			double [] params = new double [] { X.getEntry(i, 0) };
			X.setEntry(1, i, f.getValue(params));
		}
    	RealMatrix K = kernel.K(X);
		RealVector x_star = new ArrayRealVector(1);
		RealVector k_star = new ArrayRealVector(10);
		for(int i=0; i<10; i++) {
			
		}
		double x_star_covar = kernel.k(x_star, x_star);
		double noise = 0d;
		GPRegression.GPRegressionResult reg = GPRegression.predict(X, y, K, k_star, x_star_covar, noise, x_star);
	}
	
}
