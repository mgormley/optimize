package edu.jhu.hlt.util.math;

import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * A class for doing Gaussian process regression based on: 
 * 
 * 	Gaussian Processes for Machine Learning, Carl Edward Rasmussen and Chris Williams, the MIT Press, 2006
 * 
 * @author noandrews
 */

public class GPRegression {
	
	public static class GPRegressionResult {
		double mean;
		double var;
		public GPRegressionResult(double mean, double var) {
			this.mean = mean;
			this.var = var;
		}
	}
	
	public static GPRegressionResult predict(RealMatrix x,        // train inputs
			                                 RealVector y,        // train outputs
			                                 RealMatrix K,        // covar between training points
			                                 RealVector k_star,   // covar between test x_star and all training x
			                                 double x_star_covar, // self-covar of x_star
			                                 double noise,        // noise level in inputs
			                                 RealVector x_star) { // test input
		RealMatrix temp = K.subtract(MatrixUtils.createRealIdentityMatrix(K.getColumnDimension()).scalarMultiply(noise));
		CholeskyDecomposition decomp = new CholeskyDecomposition(temp);
		RealMatrix L = decomp.getL();
		RealMatrix LT = decomp.getLT();
		RealVector alpha = y.copy();
		MatrixUtils.solveLowerTriangularSystem(L, alpha);
		MatrixUtils.solveUpperTriangularSystem(LT, alpha);
		double predicted_mean = k_star.dotProduct(alpha);
		RealVector v = k_star.copy();
		MatrixUtils.solveLowerTriangularSystem(L, v);
		double predicted_var = x_star_covar - v.dotProduct(v);
		return new GPRegressionResult(predicted_mean, predicted_var);
	}
}
