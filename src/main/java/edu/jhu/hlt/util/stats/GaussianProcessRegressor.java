package edu.jhu.hlt.util.stats;

/*
 * Replacement class for GPRegression
 * 
 * @author noandrews
 */

import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class GaussianProcessRegressor implements Regressor {

	RealMatrix X;
	RealMatrix Y;
	RealMatrix K;
	RealMatrix Kinv;
	CholeskyDecomposition decomp;
	RealVector alpha;
	Kernel kernel;
	double noise;
	
	public GaussianProcessRegressor(Kernel k) {
		this.kernel = k;
	}
	
	@Override
	public void fit(RealMatrix X, RealMatrix Y) {
		K = kernel.K(X);
		assert(K.getColumnDimension() == Y.getColumnDimension()) : "dimension mismatch: " + K.getColumnDimension() + " != " + Y.getColumnDimension();
		RealMatrix temp = K.add(MatrixUtils.createRealIdentityMatrix(K.getColumnDimension()).scalarMultiply(noise));
		decomp = new CholeskyDecomposition(temp);
		RealMatrix L = decomp.getL();
		RealMatrix LT = decomp.getLT();
		RealVector alpha = Y.getColumnVector(0);
		MatrixUtils.solveLowerTriangularSystem(L, alpha);
		MatrixUtils.solveUpperTriangularSystem(LT, alpha);
	}

	@Override
	public Regression predict(RealMatrix Xstar) {
		RealMatrix means = MatrixUtils.createRealMatrix(1, Xstar.getColumnDimension());
		RealMatrix vars = MatrixUtils.createRealMatrix(1, Xstar.getColumnDimension());
		for(int i=0; i<Xstar.getColumnDimension(); i++) {
			RealVector x_star = Xstar.getColumnVector(i);
			double x_star_covar = kernel.k(x_star, x_star);
			// FIXME
			// RealVector k_star = vectorCovar(X, x_star, kernel);
			RealVector k_star = null;
			double predicted_mean = k_star.dotProduct(alpha);
			RealVector v = k_star.copy();
			MatrixUtils.solveLowerTriangularSystem(decomp.getL(), v);
			double predicted_var = x_star_covar - v.dotProduct(v);
		}
		return null;
	}
}
