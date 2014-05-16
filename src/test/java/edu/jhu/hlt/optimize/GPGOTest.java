package edu.jhu.hlt.optimize;

import static org.junit.Assert.assertEquals;

import java.awt.Color;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.junit.Test;

import com.xeiam.xchart.Chart;
import com.xeiam.xchart.ChartBuilder;
import com.xeiam.xchart.Series;
import com.xeiam.xchart.SeriesLineStyle;
import com.xeiam.xchart.SeriesMarker;
import com.xeiam.xchart.StyleManager.ChartType;
import com.xeiam.xchart.SwingWrapper;

import edu.jhu.hlt.optimize.function.Bounds;
import edu.jhu.hlt.optimize.function.ConstrainedDifferentiableFunction;
import edu.jhu.hlt.optimize.function.ConstrainedFunction;
import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.DifferentiableFunctionOpts.DifferentiableFunctionWithConstraints;
import edu.jhu.hlt.optimize.function.DifferentiableFunctionOpts.NegateFunction;
import edu.jhu.hlt.optimize.function.Function;
import edu.jhu.hlt.optimize.function.FunctionOpts;
import edu.jhu.hlt.optimize.functions.Rastrigins;
import edu.jhu.hlt.optimize.functions.SimpleCubicFunction;
import edu.jhu.hlt.optimize.functions.UnevenDecreasingMaxima;
import edu.jhu.util.Prng;
import edu.jhu.hlt.util.stats.Kernel;
import edu.jhu.hlt.util.stats.RegressionResult;
import edu.jhu.hlt.util.stats.SquaredExpKernel;
import edu.jhu.prim.vector.IntDoubleDenseVector;

public class GPGOTest {

	static Logger log = Logger.getLogger(GPGOTest.class);
	
	@Test
	public void myopicLossTest() {
		
		Prng.seed(42);
		
    	BasicConfigurator.configure();
    	Logger.getRootLogger().setLevel(Level.DEBUG);
		
    	// Parameters
    	Kernel kernel = new SquaredExpKernel(0.5, 1.0);
    	
    	//Function f = new XSquared(0);
    	Function f = new SimpleCubicFunction();
    	
    	// Training data:
    	//   X	inputs
    	//   y  ouputs
    	double [][] xs = {{-1.0}, {-0.5}, {0.3}, {0.7}, {0.9}};
		RealMatrix X = MatrixUtils.createRealMatrix(xs).transpose();
		double [] ys = new double[xs.length];
		for(int i=0; i<ys.length; i++) {
			ys[i] = f.getValue(new IntDoubleDenseVector(xs[i]));
		}
		RealVector y = new ArrayRealVector(ys);
		
		List<Number> training_x = new ArrayList<Number>();
		List<Number> training_y = new ArrayList<Number>();
		
		for(int i=0; i<X.getColumnDimension(); i++) {
			training_x.add(xs[i][0]);
			training_y.add(ys[i]);
		}

		// Initialize the GPGO instance
		double [] A = new double[1];
		double [] B = new double[1];
		A[0] = -3.0;
		B[0] = +3.0;
		Bounds bounds = new Bounds(A, B);
		ConstrainedFunction g = new FunctionOpts.FunctionWithConstraints(f, bounds);
		GPGO opt = new GPGO(g, kernel, X, y, 0d);
		
		// Estimate the GP posterior
		opt.estimatePosterior();
		
		double [] start_pt = new double[] {-1.5};
		double start_loss = opt.loss.getValue(new IntDoubleDenseVector(start_pt));
		List<Number> start_x = new ArrayList<Number>();
		List<Number> start_y = new ArrayList<Number>();
		
		start_x.add(start_pt[0]);
		start_y.add(start_loss);
		
		log.info("Initial loss = l(" + start_pt[0] + ")="+start_loss);
		
		GradientDescentWithLineSearch local_opt = new GradientDescentWithLineSearch(25);
		IntDoubleDenseVector v = new IntDoubleDenseVector(start_pt);
		local_opt.minimize(opt.loss, v);
		double yguess = opt.loss.getValue(v);
		
		log.info("xguess = " + v.get(0));
		log.info("yguess = " + yguess);
		
		// Test some other points of the loss function
		//double [] pt1 = {-1};
		//opt.loss.setPoint(pt1);
		//log.info("loss("+pt1[0]+")="+opt.loss.getValue());
		//double [] pt2 = {0};
		//opt.loss.setPoint(pt2);
		//log.info("loss("+pt2[0]+")="+opt.loss.getValue());
		
		List<Number> next_x = new ArrayList<Number>();
		next_x.add(v.get(0));
		List<Number> next_y = new ArrayList<Number>();
		next_y.add(yguess);
		
//		RealVector x_star = new ArrayRealVector(new double[] {-1.8});
//		double [] loss_grad = opt.loss.computeExpectedLossGradient(x_star);
//		
//		double eps = 1e-6;
//		RealVector x_plus_eps = x_star.copy();
//		x_plus_eps.addToEntry(0, eps);
//		RealVector x_minus_eps = x_star.copy();
//		x_minus_eps.addToEntry(0, -eps);
//		double approx_loss_grad = opt.loss.computeExpectedLoss(x_plus_eps) - opt.loss.computeExpectedLoss(x_minus_eps);
//		approx_loss_grad /= 2*eps;
		
//		log.info("loss grad = " + loss_grad[0]);
//		log.info("approx loss grad = " + approx_loss_grad);
		
		List<Number> posterior_mean = new ArrayList<Number>();
		List<Number> posterior_var = new ArrayList<Number>();
		List<Number> eloss = new ArrayList<Number>();
		
		double grid_min = A[0];
		double grid_max = B[0];
		double range = grid_max - grid_min;
		int npts = 50;
		double increment = range/(double)npts; 
		List<Number> grid = new ArrayList<Number>();
		
		// Series 3 (actual function)
		List<Number> fvals = new ArrayList<Number>();
		for(double x=grid_min; x<grid_max; x+=increment) {
			//log.info("x = " + x);
			
			grid.add(x);
			fvals.add(f.getValue(new IntDoubleDenseVector(new double[] {x})));
			RealVector x_star = new ArrayRealVector(new double[] {x});
			RegressionResult pred = opt.getRegressor().predict(new ArrayRealVector(new double[] {x}));
			double obj1 = opt.getExpectedLoss().computeExpectedLoss(new ArrayRealVector(new double[] {x}));
			double obj2 = opt.loss.getValue(new IntDoubleDenseVector(new double[] {x}));
			//double obj = 0;
			
			log.info("loss1("+x+")="+obj1);
			log.info("loss2("+x+")="+obj2);
			  
			// GP predictions
			posterior_mean.add(pred.mean);
			posterior_var.add(pred.var);
			  
			// Loss
			eloss.add(obj1);
			
			// Gradient of loss
			double [] loss_grad = opt.loss.computeExpectedLossGradient(x_star);
	
			double eps = 1e-6;
			RealVector x_plus_eps = x_star.copy();
			x_plus_eps.addToEntry(0, eps);
			RealVector x_minus_eps = x_star.copy();
			x_minus_eps.addToEntry(0, -eps);
			double approx_loss_grad = opt.loss.computeExpectedLoss(x_plus_eps) - opt.loss.computeExpectedLoss(x_minus_eps);
			approx_loss_grad /= 2*eps;
			
			log.info("loss grad = " + loss_grad[0]);
			log.info("approx loss grad = " + approx_loss_grad);
		}
		
		// Create Chart
		//Chart chart = new ChartBuilder().width(800).height(600).title("ScatterChart04").xAxisTitle("X").yAxisTitle("Y").chartType(ChartType.Scatter).build();
		Chart chart = new ChartBuilder().width(800).height(600).title("ScatterChart04").xAxisTitle("X").yAxisTitle("Y").chartType(ChartType.Line).build();
		
		// Customize Chart
		chart.getStyleManager().setChartTitleVisible(false);
		chart.getStyleManager().setLegendVisible(true);
		chart.getStyleManager().setAxisTitlesVisible(false);
		 
		// Series 0 (observations)
		Series series0 = chart.addSeries("Observations", training_x, training_y);
		series0.setLineStyle(SeriesLineStyle.NONE);
		series0.setMarker(SeriesMarker.SQUARE);
		series0.setMarkerColor(Color.BLACK);
		
		// Series 1 (GP posterior)
		Series series1 = chart.addSeries("Posterior", grid, posterior_mean, posterior_var);
		series1.setMarkerColor(Color.RED);
		series1.setMarker(SeriesMarker.NONE);
		
		// Series 2 (expected loss)
		Series series2 = chart.addSeries("Expected loss", grid, eloss);
		series2.setMarkerColor(Color.GREEN);
		series2.setMarker(SeriesMarker.NONE);
		
		Series series3 = chart.addSeries("Function", grid, fvals);
		series3.setMarkerColor(Color.BLACK);
		series3.setMarker(SeriesMarker.NONE);
		
		Series series4 = chart.addSeries("Next eval", next_x, next_y);
		series4.setMarker(SeriesMarker.TRIANGLE_DOWN);
		series4.setMarkerColor(Color.CYAN);
		
		Series series5 = chart.addSeries("Start of local search", start_x, start_y);
		series5.setMarker(SeriesMarker.TRIANGLE_DOWN);
		series5.setMarkerColor(Color.GRAY);
		
		chart.getStyleManager().setYAxisMin(-3);
		chart.getStyleManager().setYAxisMax(3);
		 
		chart.getStyleManager().setXAxisMin(A[0]);
		chart.getStyleManager().setXAxisMax(B[0]);
		
		// Show it
	    new SwingWrapper(chart).displayChart();
	    
	    log.info("done!");
		
	}
	
	// This unit test takes too long. Run via main().
	// @Test
	public void Rastrigins() {

		BasicConfigurator.configure();
		Logger.getRootLogger().setLevel(Level.DEBUG);

		Prng.seed(42);

		int D;
		int maxiter = 100;

		// Low dimensions
		double noise = 0.01;
		D = 2;

		DifferentiableFunction f = new Rastrigins(D);
		// The rastrigin optimum is at vec(0)

		// Optimization bounds: 5.12 xi 5.12
		double [] L = new double[D];
		double [] U = new double[D];
		double [] start = new double[D];
		for(int i=0; i<D; i++) {
			L[i] = -5.12;
			U[i] = 5.12;
			start[i] = Prng.nextDouble();
		}
		log.info("starting pt = ("+start[0]+", "+start[1]+")");
		Bounds b = new Bounds(L, U);

		ConstrainedDifferentiableFunction g = new DifferentiableFunctionWithConstraints(f, b);
		
		SquaredExpKernel kernel = new SquaredExpKernel();
		GPGO opt = new GPGO(g, kernel, noise, maxiter);

		IntDoubleDenseVector opt_point = new IntDoubleDenseVector(start);
		opt.minimize(g, opt_point);

		double opt_val = f.getValue(opt_point);

		log.info("found opt val = " + opt_val);
		assertEquals(0, opt_val, 0.1);
	}

    // This unit test takes too long. Run via main().
    // @Test
	public void UnevenDecreasingMaxima() {
		
		BasicConfigurator.configure();
		Logger.getRootLogger().setLevel(Level.DEBUG);
		
		UnevenDecreasingMaxima g = new UnevenDecreasingMaxima();
		Function f = new NegateFunction(g);
		
		double grid_min = 0.0;
		double grid_max = 1.0;
		double range = grid_max - grid_min;
		int npts = 500;
		int npts2 = 100;
		double increment = range/(double)npts;
		double increment2 = range/(double)npts2;
		
		List<Number> grid = new ArrayList<Number>();
		List<Number> fvals = new ArrayList<Number>();
		
		for(double x=grid_min; x<grid_max; x+=increment) {	
			double y = f.getValue(new IntDoubleDenseVector(new double[] {x}));
			grid.add(x);
			fvals.add(y);
		}
		
		// Initialize GPGO
		double [] A = new double[1];
		double [] B = new double[1];
		A[0] = 0;
		B[0] = 1;
		Bounds bounds = new Bounds(A, B);
		ConstrainedFunction h = new FunctionOpts.FunctionWithConstraints(f, bounds);
		
		// These parameters are crucial
		Kernel kernel = new SquaredExpKernel(10, 1);
		
		GPGO opt = new GPGO(h, kernel, 0.001, 10);
		opt.setSearchParam(30000, 5);
		
		// Uncomment these two to just run GPGO normally
		//opt.minimize();
		//System.exit(0);
		
		// Do some iterations of GPGO
		opt.setInitialPoint();
		opt.setInitialPoint();
		opt.setInitialPoint();
		
		for(int iter=0; iter<1; iter++) {
			
			// Observations
			List<Number> xs = new ArrayList<Number>();
			List<Number> ys = new ArrayList<Number>();
			
			for(int i=0; i<opt.X.getColumnDimension(); i++) {
				xs.add(opt.X.getColumnVector(i).getEntry(0));
				ys.add(opt.y.getEntry(i));
			}
			
			RealVector min_vec = opt.doIterNoUpdate(iter, true);
			List<Number> min = new ArrayList<Number>();
			min.add(min_vec.getEntry(0));
			List<Number> loss_at_min = new ArrayList<Number>();
			loss_at_min.add(opt.loss.computeExpectedLoss(min_vec));
			
			Chart chart = new ChartBuilder().width(800).height(600).title("iter"+iter).xAxisTitle("X").yAxisTitle("Y").chartType(ChartType.Line).build();
			
			// Customize Chart
			chart.getStyleManager().setChartTitleVisible(true);
			chart.getStyleManager().setLegendVisible(true);
			chart.getStyleManager().setAxisTitlesVisible(false);
	 
			List<Number> posterior_mean = new ArrayList<Number>();
			List<Number> posterior_var = new ArrayList<Number>();
			List<Number> grid2 = new ArrayList<Number>();
			List<Number> ls = new ArrayList<Number>();
			
			for(double x=grid_min; x<grid_max; x+=increment2) {
				grid2.add(x);
				RegressionResult pred = opt.getRegressor().predict(new ArrayRealVector(new double[] {x}));
				posterior_mean.add(pred.mean);
				posterior_var.add(pred.var);
				
				double l = opt.loss.getValue(new IntDoubleDenseVector(new double[] {x}));
				//log.info("loss("+x+")="+l);
				ls.add(l);
			}
			
			// Series 0: True function
			Series series0 = chart.addSeries("True function", grid, fvals);
			series0.setMarker(SeriesMarker.NONE);
		
			// Series 1: Posterior
			Series series1 = chart.addSeries("Posterior", grid2, posterior_mean, posterior_var);
			series1.setMarker(SeriesMarker.NONE);
			
			// Series 2: Loss
			Series series2 = chart.addSeries("Loss", grid2, ls);
			series2.setMarker(SeriesMarker.NONE);
			
			// Series 3: Observations
			Series series3 = chart.addSeries("Observations", xs, ys);
			series3.setLineStyle(SeriesLineStyle.NONE);
			
			// Series 4: Chosen point
			Series series4 = chart.addSeries("Next evaluation", min, loss_at_min);
			series4.setLineStyle(SeriesLineStyle.NONE);
			
			chart.getStyleManager().setYAxisMin(-1.0);
			chart.getStyleManager().setYAxisMax(0.5);
		 
			chart.getStyleManager().setXAxisMin(0);
			chart.getStyleManager().setXAxisMax(1);
		
			new SwingWrapper(chart).displayChart();
			
			// Update observations for the next iteration
			opt.updateObservations(min_vec, h.getValue(new IntDoubleDenseVector(min_vec.toArray())));
		}
	}

    // This unit test takes too long. Run via main().
    // @Test
	public void UnevenDecreasingMaximaOptimizedByGD() {
		
		Prng.seed(54);
		
		BasicConfigurator.configure();
		Logger.getRootLogger().setLevel(Level.DEBUG);
		
		//UnevenDecreasingMaxima g = new UnevenDecreasingMaxima();
		//Function f = new FunctionOpts.NegateFunction(g);
		
		Function f = new UnevenDecreasingMinimaOptimizedByGD(100);
		
		double grid_min = 0.03;
		double grid_max = 0.97;
		double range = grid_max - grid_min;
		int npts = 500;
		int npts2 = 100;
		double increment = range/(double)npts;
		double increment2 = range/(double)npts2;
		
		List<Number> grid = new ArrayList<Number>();
		List<Number> fvals = new ArrayList<Number>();
		
		for(double x=grid_min; x<grid_max; x+=increment) {	
			double y = f.getValue(new IntDoubleDenseVector(new double[] {x}));
			grid.add(x);
			fvals.add(y);
		}
		
		// Initialize GPGO
		double [] A = new double[1];
		double [] B = new double[1];
		A[0] = 0;
		B[0] = 1;
		Bounds bounds = new Bounds(A, B);
		ConstrainedFunction h = new FunctionOpts.FunctionWithConstraints(f, bounds);
		
		// These parameters are crucial
		Kernel kernel = new SquaredExpKernel(100, 0.001);
		
		GPGO opt = new GPGO(h, kernel, 0.001, 10);
		opt.setSearchParam(30000, 5);
		
		// Uncomment these two to just run GPGO normally
		//opt.minimize();
		//System.exit(0);
		
		// Do some iterations of GPGO
		opt.setInitialPoint();
		opt.setInitialPoint();
		//opt.setInitialPoint();
		
		for(int iter=0; iter<2; iter++) {
			
			// Observations
			List<Number> xs = new ArrayList<Number>();
			List<Number> ys = new ArrayList<Number>();
			
			for(int i=0; i<opt.X.getColumnDimension(); i++) {
				xs.add(opt.X.getColumnVector(i).getEntry(0));
				ys.add(opt.y.getEntry(i));
			}
			
			RealVector min_vec = opt.doIterNoUpdate(iter, true);
			List<Number> min = new ArrayList<Number>();
			min.add(min_vec.getEntry(0));
			List<Number> loss_at_min = new ArrayList<Number>();
			loss_at_min.add(opt.loss.computeExpectedLoss(min_vec));
			
			Chart chart = new ChartBuilder().width(800).height(600).title("iter"+iter).xAxisTitle("X").yAxisTitle("Y").chartType(ChartType.Line).build();
			
			// Customize Chart
			chart.getStyleManager().setChartTitleVisible(true);
			chart.getStyleManager().setLegendVisible(true);
			chart.getStyleManager().setAxisTitlesVisible(false);
	 
			List<Number> posterior_mean = new ArrayList<Number>();
			List<Number> posterior_var = new ArrayList<Number>();
			List<Number> grid2 = new ArrayList<Number>();
			List<Number> ls = new ArrayList<Number>();
			
			for(double x=grid_min; x<grid_max; x+=increment2) {
				grid2.add(x);
				RegressionResult pred = opt.getRegressor().predict(new ArrayRealVector(new double[] {x}));
				posterior_mean.add(pred.mean);
				posterior_var.add(pred.var);
				
				double l = opt.loss.getValue(new IntDoubleDenseVector(new double[] {x}));
				//log.info("loss("+x+")="+l);
				ls.add(l);
			}
			
			String obs_path = new String("/home/nico/gp-opt/papers/nips2013/plots/gpgo_func_1d_obs_iter_"+iter+".dat");
			String mean_path = new String("/home/nico/gp-opt/papers/nips2013/plots/gpgo_func_1d_mean_iter_"+iter+".dat");
			String var_path = new String("/home/nico/gp-opt/papers/nips2013/plots/gpgo_func_1d_var_iter_"+iter+".dat");
			
			try {
				Writer writer1 = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(obs_path), "utf-8"));
				Writer writer2 = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(mean_path), "utf-8"));
				Writer writer3 = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(var_path), "utf-8"));
			    
				for(int i=0; i<xs.size(); i++) {
					writer1.write(xs.get(i) + " " + ys.get(i) + "\n");
				}
				
				for(int i=0; i<grid2.size(); i++) {
					writer2.write(grid2.get(i) + " " + posterior_mean.get(i) + "\n");
					writer3.write(grid2.get(i) + " " + posterior_var.get(i) + "\n");
				}
				
				writer1.close();
				writer2.close();
				writer3.close();
			} catch (IOException ex){
			    // handle me
			} 
			
			// Series 0: True function
			Series series0 = chart.addSeries("True function", grid, fvals);
			series0.setMarker(SeriesMarker.NONE);
		
			// Series 1: Posterior
			Series series1 = chart.addSeries("Posterior", grid2, posterior_mean, posterior_var);
			series1.setMarker(SeriesMarker.NONE);
			
			// Series 2: Loss
			Series series2 = chart.addSeries("Loss", grid2, ls);
			series2.setMarker(SeriesMarker.NONE);
			
			// Series 3: Observations
			Series series3 = chart.addSeries("Observations", xs, ys);
			series3.setLineStyle(SeriesLineStyle.NONE);
			
			// Series 4: Chosen point
			Series series4 = chart.addSeries("Next evaluation", min, loss_at_min);
			series4.setLineStyle(SeriesLineStyle.NONE);
			
			chart.getStyleManager().setYAxisMin(-1.0);
			chart.getStyleManager().setYAxisMax(0.5);
		 
			chart.getStyleManager().setXAxisMin(0);
			chart.getStyleManager().setXAxisMax(1);
		
			new SwingWrapper(chart).displayChart();
			
			// Update observations for the next iteration
			opt.updateObservations(min_vec, h.getValue(new IntDoubleDenseVector(min_vec.toArray())));
		}
	}
	
	public static void main(String [] args) {
		GPGOTest tester = new GPGOTest();
		tester.Rastrigins();
		tester.UnevenDecreasingMaxima();
		tester.UnevenDecreasingMaximaOptimizedByGD();
	}
	
}
