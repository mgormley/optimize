package edu.jhu.hlt.util.math;

import java.awt.Color;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.log4j.Logger;

import com.xeiam.xchart.Chart;
import com.xeiam.xchart.ChartBuilder;
import com.xeiam.xchart.Series;
import com.xeiam.xchart.SeriesMarker;
import com.xeiam.xchart.SwingWrapper;
import com.xeiam.xchart.StyleManager.ChartType;

/**
 * A class for doing Gaussian process regression based on: 
 * 
 * 	Gaussian Processes for Machine Learning, Carl Edward Rasmussen and Chris Williams, the MIT Press, 2006
 * 
 * Wishlist:
 * 	- Rank 1 updates (update an existing Cholesky decomposition in O(n^2) after observing a new x) 
 * 
 * @author noandrews
 */

public class GPRegression {
	
	static Logger log = Logger.getLogger(GPRegression.class);
	
	public static class RegressionResult {
		public double mean;
		public double var;
		public RegressionResult(double mean, double var) {
			this.mean = mean;
			this.var = var;
		}
	}
	
	// TODO: this should implement a generic "Regressor" interface
	public static class GPRegressor {
		RealMatrix X;
		CholeskyDecomposition decomp;
		RealVector alpha;
		Kernel kernel;
		double noise;
		public GPRegressor(RealMatrix X, CholeskyDecomposition decomp, RealVector alpha, Kernel kernel, double noise) {
			this.X = X;
			this.decomp = decomp;
			this.alpha = alpha;
			this.kernel = kernel;
			this.noise = noise;
		}
		
		public RegressionResult predict(RealVector x_star) {
			double x_star_covar = kernel.k(x_star, x_star);
			RealVector k_star = vectorCovar(X, x_star, kernel);
			//for(int k=0; k<k_star.getDimension(); k++) {
			//	log.info("k_star["+k+"]="+k_star.getEntry(k));
			//}
			double predicted_mean = k_star.dotProduct(alpha);
			RealVector v = k_star.copy();
			MatrixUtils.solveLowerTriangularSystem(decomp.getL(), v);
			double predicted_var = x_star_covar - v.dotProduct(v);
			//assert(predicted_var > 0) : "variance not strictly positive";
			return new RegressionResult(predicted_mean, predicted_var+noise);
		}
		
		/*
		 * wrt x_star
		 */
		public void computeMeanGradient(RealVector x_star, double [] g) {
			for(int k=0; k<g.length; k++) {
				g[k] = 0;
			}
			double [] kernel_grad = new double[g.length];
			for(int i=0; i<decomp.getL().getColumnDimension(); i++) { // number of training instances
				kernel.grad_k(X.getColumnVector(i), x_star, kernel_grad);
				for(int k=0; k<g.length; k++) {
					g[k] += kernel_grad[k] * alpha.getEntry(i);
				}
			}
		}
		
		//FIXME: speed up this computation
		public void computeCovarGradient(RealVector x_star, double [] g) {
			RealVector k_star = vectorCovar(X, x_star, kernel);
			RealMatrix G = vectorCovarGradient(X, x_star, kernel); // this could be a problem space-wise
//			for(int i=0; i<G.getRowDimension(); i++) {
//				for(int j=0; j<G.getColumnDimension(); j++) {
//					log.info("G "+i+" "+j+" = "+G.getEntry(i, j));
//				}
//			}
			RealMatrix K_inverse = decomp.getSolver().getInverse();
			RealMatrix temp = K_inverse.multiply(G);
			RealMatrix k_star_T = MatrixUtils.createRealMatrix(k_star.getDimension(), 1);
			k_star_T.setColumnVector(0, k_star);
			k_star_T = k_star_T.transpose();
			RealMatrix result = k_star_T.multiply(temp);
			//log.info("ncol = " + result.getColumnDimension());
			//log.info("nrow = " + result.getRowDimension());
			for(int i=0; i<g.length; i++) {
				g[i] = -2*result.getEntry(0, i);
			}
		}
		
		public RealMatrix getL() {
			return decomp.getL();
		}
		
		public RealVector getAlpha() {
			return alpha;
		}
		
		public RealVector getInput(int i) {
			return X.getColumnVector(i);
		}
	}
	
	public static RealVector vectorCovar(RealMatrix X, RealVector x, Kernel kernel) {
		RealVector k_star = new ArrayRealVector(X.getColumnDimension());
		for(int i=0; i<X.getColumnDimension(); i++) {
			k_star.setEntry(i, kernel.k(X.getColumnVector(i), x));
		}
		return k_star;
	}
	
	/**
	 * dk* / dx
	 * 
	 * Returns matrix with X.getColumnDimension() rows and x.getDimension() columns
	 */
	public static RealMatrix vectorCovarGradient(RealMatrix X, RealVector x, Kernel kernel) {
		RealMatrix G = MatrixUtils.createRealMatrix(x.getDimension(), X.getColumnDimension());
		for(int i=0; i<X.getColumnDimension(); i++) {
			double [] g = new double[x.getDimension()];
			kernel.grad_k(X.getColumnVector(i), x, g);
			G.setColumn(i, g);
		}
		return G.transpose();
	}
	
	public static GPRegressor trainRegressor(RealMatrix X, // train inputs
								             RealVector y, // train outputs
								             Kernel kernel,
								             double noise  // noise level in inputs 
								             ) {
		RealMatrix K = kernel.K(X);
		assert(K.getColumnDimension() == y.getDimension()) : "dimension mismatch: " + K.getColumnDimension() + " != " + y.getDimension();
		RealMatrix temp = K.add(MatrixUtils.createRealIdentityMatrix(K.getColumnDimension()).scalarMultiply(noise));
		CholeskyDecomposition decomp = new CholeskyDecomposition(temp);
		RealMatrix L = decomp.getL();
		RealMatrix LT = decomp.getLT();
		RealVector alpha = y.copy();
		MatrixUtils.solveLowerTriangularSystem(L, alpha);
		MatrixUtils.solveUpperTriangularSystem(LT, alpha);
		return new GPRegressor(X, decomp, alpha, kernel, noise);
	}
	
	public static RegressionResult predict(RealMatrix x,        // train inputs
			                               RealVector y,        // train outputs
			                               RealMatrix K,        // covar between training points
			                               RealVector k_star,   // covar between test x_star and all training x
			                               double x_star_covar, // self-covar of x_star
			                               double noise,        // noise level in inputs
			                               RealVector x_star) { // test input
		RealMatrix temp = K.add(MatrixUtils.createRealIdentityMatrix(K.getColumnDimension()).scalarMultiply(noise));
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
		return new RegressionResult(predicted_mean, predicted_var);
	}
	
	public static void main(String [] args) {
		// Parameters
    	Kernel kernel = new SquaredExpKernel(1d, 1d);
    	double noise = 0d;
    	
    	// Training data
    	double [][] xs = {{-3},{-2},{-1}, {1}, {2}, {3}};
		RealMatrix X = MatrixUtils.createRealMatrix(xs).transpose();
		double [] ys = new double[xs.length];
		for(int i=0; i<ys.length; i++) {
			ys[i] = xs[i][0]*xs[i][0];
		}
		RealVector y = new ArrayRealVector(ys);

		GPRegressor reg = GPRegression.trainRegressor(X, y, kernel, noise);
		
		// generates data
		List<Number> xData1 = new ArrayList<Number>();
		List<Number> yData1 = new ArrayList<Number>();
		List<Number> errorBars1 = new ArrayList<Number>();
		for (int i=0; i<X.getColumnDimension(); i++) {
		  xData1.add(xs[i][0]);
		  RegressionResult pred = reg.predict(X.getColumnVector(i));
		  yData1.add(pred.mean);
		  errorBars1.add(pred.var);
		}
		
		double [][] xs2 = {{-6},{-4},{-2.5},{-1.5}, {-0.5}, {0.5}, {1.5}, {2.5}, {4}, {-6}};
		RealMatrix X2 = MatrixUtils.createRealMatrix(xs2).transpose();
		List<Number> xData2 = new ArrayList<Number>();
		List<Number> yData2 = new ArrayList<Number>();
		List<Number> errorBars2 = new ArrayList<Number>();
		for (int i=0; i<X2.getColumnDimension(); i++) {
		  xData2.add(xs2[i][0]);
		  RegressionResult pred = reg.predict(X2.getColumnVector(i));
		  yData2.add(pred.mean);
		  errorBars2.add(pred.var);
		}
		
		// Create Chart
		Chart chart = new ChartBuilder().width(800).height(600).title("ScatterChart04").xAxisTitle("X").yAxisTitle("Y").chartType(ChartType.Scatter).build();
		 
		// Customize Chart
		chart.getStyleManager().setChartTitleVisible(false);
		chart.getStyleManager().setLegendVisible(false);
		chart.getStyleManager().setAxisTitlesVisible(false);
		 
		// Series 1 (observations)
		Series series1 = chart.addSeries("Observations", xData1, yData1, errorBars1);
		series1.setMarkerColor(Color.RED);
		series1.setMarker(SeriesMarker.SQUARE);
		
		// Series 2 (predictions)
		Series series2 = chart.addSeries("Predictions", xData2, yData2, errorBars2);
		series2.setMarkerColor(Color.GREEN);
		series2.setMarker(SeriesMarker.DIAMOND);
		
		// Show it
	    new SwingWrapper(chart).displayChart();
	}
}
