package edu.jhu.hlt.util.stats;

/**
 * Apache Math provides similar functionality. This class exists primarily
 * to facilitate benchmarking of other regression classes.
 * 
 * @author noandrews
 */

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.log4j.Logger;

import edu.jhu.hlt.optimize.functions.Line;

public class LinearRegressor implements Regressor {

	double var = 0.1;
	static Logger log = Logger.getLogger(LinearRegressor.class);
	RealMatrix beta;
	
	public LinearRegressor() {}
	public LinearRegressor(double var) {}
	
	private boolean sanity(RealMatrix X, RealMatrix Y, RealMatrix lambda) {
		if(Y.getColumnDimension() != lambda.getColumnDimension()) return false;
		return true;
	}
	
	private RealMatrix getL2(RealMatrix X) {
		RealMatrix lambda = MatrixUtils.createRealMatrix(1, X.getColumnDimension());
		for(int i=0; i<lambda.getColumnDimension(); i++) {
			lambda.setEntry(0, i, var);
		}
		return lambda;
	}
	
	public RealMatrix getBeta() {
		return beta;
	}
	
	@Override
	public void fit(RealMatrix X, RealMatrix Y) {
		RealMatrix lambda = getL2(X);
		RealMatrix identity = MatrixUtils.createRealIdentityMatrix(X.getColumnDimension());
		identity.multiply(lambda);
		RealMatrix dataCopy = X.copy();
		RealMatrix transposeData = dataCopy.transpose();
		RealMatrix norm = transposeData.multiply(X);
		RealMatrix circular = norm.add(identity);
		LUDecomposition decomp = new LUDecomposition(circular);
		RealMatrix circularInverse = decomp.getSolver().getInverse();
		RealMatrix former = circularInverse.multiply(X.transpose());
		beta = former.multiply(Y);
		log.info("beta dims = " + beta.getRowDimension() + ", " + beta.getColumnDimension());
	}

	@Override
	public Regression predict(RealMatrix X) {
		RealMatrix predict = X.transpose().multiply(beta);
		return new Regression(predict, null);
	}

	public static void main(String [] args) {
		
		LinearRegressor reg = new LinearRegressor();
		Line line = new Line(3.14, 0.0);
		NormalDistribution noise = new NormalDistribution(0, 1);
		
		// Create train data
		int num_train = 10;
		RealMatrix X = MatrixUtils.createRealMatrix(1, num_train);
		RealMatrix Y = MatrixUtils.createRealMatrix(1, num_train);
		
		for(int i=0; i<num_train; i++) {
			double x = (double)i;
			double y = line.getValue(x) + noise.sample();
			X.setEntry(0, i, x);
			Y.setEntry(0, i, y);
		}
		
		X = X.transpose();
		Y = Y.transpose();
		
		// Fit regression model
		reg.fit(X, Y);
		
		// Print parameters
		log.info("beta = " + reg.getBeta());
		
		// Create test data
		int num_test = 10;
		RealMatrix Xstar = MatrixUtils.createRealMatrix(1, num_test);
		RealMatrix Ystar = MatrixUtils.createRealMatrix(1, num_test);
		
		for(int i=0; i<num_test; i++) {
			double x = (double)i + num_train;
			double y = line.getValue(x);
			Xstar.setEntry(0, i, x);
			Ystar.setEntry(0, i, y); // note: noise free
		}
		
		// Predict on test data
		Regression res = reg.predict(Xstar);
		
		// Print the results
		RealMatrix Yhat = res.getPredictions();
		
		log.info("Yhat: " + Yhat);
		
		// Evaluate
		log.info("MSE = " + res.meanSquaredError(Ystar.transpose()));
	}
}
