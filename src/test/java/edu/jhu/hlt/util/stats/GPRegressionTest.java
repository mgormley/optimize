package edu.jhu.hlt.util.stats;

import static org.junit.Assert.*;

import java.awt.Color;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.junit.Test;

import com.xeiam.xchart.Chart;
import com.xeiam.xchart.ChartBuilder;
import com.xeiam.xchart.Series;
import com.xeiam.xchart.SeriesMarker;
import com.xeiam.xchart.StyleManager.ChartType;
import com.xeiam.xchart.SwingWrapper;

import edu.jhu.hlt.optimize.function.Function;
import edu.jhu.hlt.optimize.functions.Friedman;
import edu.jhu.hlt.optimize.functions.XSquared;
import edu.jhu.hlt.util.stats.GPRegression;
import edu.jhu.hlt.util.stats.Kernel;
import edu.jhu.hlt.util.stats.SquaredExpKernel;
import edu.jhu.hlt.util.stats.GPRegression.GPRegressor;
import edu.jhu.prim.util.Prng;
import edu.jhu.prim.vector.IntDoubleDenseVector;

public class GPRegressionTest {

	static Logger log = LoggerFactory.getLogger(GPRegressionTest.class);

	public double [][] uniformPoints(int N, int D) {
		double [][] ret = new double[N][];
		for(int i=0; i<N; i++) {
			ret[i] = new double[D];
			for(int j=0; j<D; j++) {
				ret[i][j] = Prng.nextDouble();
			}
		}
		return ret;
	}
	
	@Test
	public void hyperparameterInference() {
		
		Function f = new Friedman();
		
		// Sample some points at which to evaluate the function
		int N = 100;
		int D = 5;
		RealMatrix X      = MatrixUtils.createRealMatrix(uniformPoints(N,D)).transpose();
		RealMatrix X_star = MatrixUtils.createRealMatrix(uniformPoints(N,D)).transpose();
		
		double [] ys = new double[X.getColumnDimension()];
		double [] ys_eval = new double[X_star.getColumnDimension()];
		for(int i=0; i<N; i++) {
			ys[i] = f.getValue(new IntDoubleDenseVector(X.getColumn(i)));
			ys_eval[i] = f.getValue(new IntDoubleDenseVector(X_star.getColumn(i)));
		}
		
		RealVector y = new ArrayRealVector(ys);
		RealVector y_star = new ArrayRealVector(ys_eval);
		
		Kernel kernel = new SquaredExpKernel();
		GPRegressor reg = GPRegression.computePosterior(X, y, kernel, 0.1);
		
		// Compute initial RMSE
		double RMSE = 0;
		for(int i=0; i<N; i++) {
			RegressionResult r = reg.predict(X_star.getColumnVector(i));
			double y_hat = r.mean;
			RMSE += Math.sqrt(Math.pow(y_star.getEntry(i)-y_hat,2)/N);
		}
		log.info("RMSE before training: " + RMSE);
		
		// Train the hyper-parameters
		
		
		// Compute final RMSE
		RMSE = 0;
		for(int i=0; i<N; i++) {
			RegressionResult r = reg.predict(X_star.getColumnVector(i));
			double y_hat = r.mean;
			RMSE += Math.sqrt(Math.pow(y_star.getEntry(i)-y_hat,2)/N);
		}
		log.info("RMSE before training: " + RMSE);
	}
	
	@Test
	public void TwoDGradientTest() {
    	// Parameters
    	Kernel kernel = new SquaredExpKernel(1d, 1d);
    	double noise = 0d;
    	
    	// Training data
    	double [][] xs = {{-3,1},{-2,2},{-1,3}, {1,4}, {2,5}, {3,6}};
		RealMatrix X = MatrixUtils.createRealMatrix(xs).transpose();
		double [] ys = new double[xs.length];
		for(int i=0; i<ys.length; i++) {
			ys[i] = Math.pow(xs[i][0]-xs[i][1],2);
		}
		RealVector y = new ArrayRealVector(ys);

		GPRegressor reg = GPRegression.computePosterior(X, y, kernel, noise);
		
		double [] g = new double[xs[0].length];
		double eps = 1e-6;
		RealVector x_star = new ArrayRealVector(new double[] {1.5, 1.5});
		RealVector x_star_plus_eps = x_star.copy();
		RealVector x_star_minus_eps = x_star.copy();
		x_star_plus_eps.addToEntry(0, eps);
		x_star_minus_eps.addToEntry(0, -eps);
		reg.computeMeanGradient(x_star, g);
		double approx_g = reg.predict(x_star_plus_eps).mean - reg.predict(x_star).mean;
		approx_g /= eps;
		log.info(""+g[0]);
		log.info(""+approx_g);
		
		assertEquals(g[0], approx_g, 1e-3);
		
		reg.computeCovarGradient(x_star, g);
		approx_g = reg.predict(x_star_plus_eps).var - reg.predict(x_star).var;
		approx_g /= eps;
		double approx_g_2 = reg.predict(x_star_plus_eps).var - reg.predict(x_star_minus_eps).var;
		approx_g_2 /= 2*eps;
		log.info("exact = " + g[0]);
		log.info("approx = " + approx_g);
		log.info("approx 2 = " + approx_g_2);
		assertEquals(g[0], approx_g, 1e-3);
		
	}
	
	@Test
	public void OneDGradientTest() {
    	// Parameters
    	Kernel kernel = new SquaredExpKernel(1d, 1d);
    	Function f = new XSquared();
    	double noise = 0d;
    	
    	// Training data
    	double [][] xs = {{-3},{-2},{-1}, {1}, {2}, {3}};
		RealMatrix X = MatrixUtils.createRealMatrix(xs).transpose();
		double [] ys = new double[xs.length];
		for(int i=0; i<ys.length; i++) {
			ys[i] = f.getValue(new IntDoubleDenseVector(xs[i]));
		}
		RealVector y = new ArrayRealVector(ys);

		GPRegressor reg = GPRegression.computePosterior(X, y, kernel, noise);
		
		double [] g = new double[1];
		double eps = 1e-6;
		RealVector x_star = new ArrayRealVector(new double[] {1.5});
		RealVector x_star_plus_eps = x_star.copy();
		RealVector x_star_minus_eps = x_star.copy();
		x_star_plus_eps.addToEntry(0, eps);
		x_star_minus_eps.addToEntry(0, -eps);
		reg.computeMeanGradient(x_star, g);
		double approx_g = reg.predict(x_star_plus_eps).mean - reg.predict(x_star).mean;
		approx_g /= eps;
		log.info(""+g[0]);
		log.info(""+approx_g);
		
		assertEquals(g[0], approx_g, 1e-3);
		
		reg.computeCovarGradient(x_star, g);
		approx_g = reg.predict(x_star_plus_eps).var - reg.predict(x_star).var;
		approx_g /= eps;
		double approx_g_2 = reg.predict(x_star_plus_eps).var - reg.predict(x_star_minus_eps).var;
		approx_g_2 /= 2*eps;
		log.info("exact = " + g[0]);
		log.info("approx = " + approx_g);
		log.info("approx 2 = " + approx_g_2);
		assertEquals(g[0], approx_g, 1e-3);
	}
	
	@Test
	public void regressionTest() {
    	// Parameters
    	Kernel kernel = new SquaredExpKernel(1d, 1d);
    	Function f = new XSquared();
    	double noise = 0d;
    	
    	// Training data
    	double [][] xs = {{-3},{-2},{-1}, {1}, {2}, {3}};
		RealMatrix X = MatrixUtils.createRealMatrix(xs).transpose();
		double [] ys = new double[xs.length];
		for(int i=0; i<ys.length; i++) {
			ys[i] = f.getValue(new IntDoubleDenseVector(xs[i]));
		}
		RealVector y = new ArrayRealVector(ys);

		GPRegressor reg = GPRegression.computePosterior(X, y, kernel, noise);
		
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
		
		double [][] xs2 = {{-4},{-2.5},{-1.5}, {-0.5}, {0.5}, {1.5}, {2.5}, {4}};
		RealMatrix X2 = MatrixUtils.createRealMatrix(xs2).transpose();
		List<Number> xData2 = new ArrayList<Number>();
		List<Number> yData2 = new ArrayList<Number>();
		List<Number> errorBars2 = new ArrayList<Number>();
		for (int i=0; i<X.getColumnDimension(); i++) {
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
	    
	    log.info("done!");
	}
	
}
