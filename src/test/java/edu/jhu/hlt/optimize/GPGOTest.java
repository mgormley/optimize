package edu.jhu.hlt.optimize;

import static org.junit.Assert.assertEquals;

import java.awt.Color;
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
import com.xeiam.xchart.SwingWrapper;
import com.xeiam.xchart.StyleManager.ChartType;

import edu.jhu.hlt.optimize.GPGO.ExpectedMyopicLoss;
import edu.jhu.hlt.util.Prng;
import edu.jhu.hlt.util.math.Kernel;
import edu.jhu.hlt.util.math.SquaredExpKernel;
import edu.jhu.hlt.util.math.GPRegression.RegressionResult;

public class GPGOTest {

	static Logger log = Logger.getLogger(GPGOTest.class);
	
	@Test
	public void myopicLossTest() {
		
		Prng.seed(42);
		
    	BasicConfigurator.configure();
    	Logger.getRootLogger().setLevel(Level.DEBUG);
		
    	// Parameters
    	Kernel kernel = new SquaredExpKernel(0.5, 0.1);
    	
    	//Function f = new XSquared(0);
    	Function f = new Franks();
    	
    	// Training data:
    	//   X	inputs
    	//   y  ouputs
    	double [][] xs = {{-1.0}, {-0.5}, {0.3}, {0.7}, {0.9}};
		RealMatrix X = MatrixUtils.createRealMatrix(xs).transpose();
		double [] ys = new double[xs.length];
		for(int i=0; i<ys.length; i++) {
			ys[i] = f.getValue(xs[i]);
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
		opt.loss.setPoint(start_pt);
		double start_loss = opt.loss.getValue();
		List<Number> start_x = new ArrayList<Number>();
		List<Number> start_y = new ArrayList<Number>();
		
		start_x.add(start_pt[0]);
		start_y.add(start_loss);
		
		log.info("Initial loss = l(" + start_pt[0] + ")="+start_loss);
		
		ConstrainedGradientDescentWithLineSearch local_opt = new ConstrainedGradientDescentWithLineSearch(25);
		local_opt.minimize(opt.loss, start_pt);
		double [] xguess = opt.loss.getPoint();
		double yguess = opt.loss.getValue();
		
		log.info("xguess = " + xguess[0]);
		log.info("yguess = " + yguess);
		
		// Test some other points of the loss function
		//double [] pt1 = {-1};
		//opt.loss.setPoint(pt1);
		//log.info("loss("+pt1[0]+")="+opt.loss.getValue());
		//double [] pt2 = {0};
		//opt.loss.setPoint(pt2);
		//log.info("loss("+pt2[0]+")="+opt.loss.getValue());
		
		List<Number> next_x = new ArrayList<Number>();
		next_x.add(xguess[0]);
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
			fvals.add(f.getValue(new double[] {x}));
			RealVector x_star = new ArrayRealVector(new double[] {x});
			RegressionResult pred = opt.getRegressor().predict(new ArrayRealVector(new double[] {x}));
			double obj1 = opt.getExpectedLoss().computeExpectedLoss(new ArrayRealVector(new double[] {x}));
			double obj2 = opt.loss.getValue(new double[] {x});
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
	
	@Test
	public void Rastrigins() {

		BasicConfigurator.configure();
		Logger.getRootLogger().setLevel(Level.DEBUG);

		Prng.seed(42);

		int D;
		int maxiter = 100;

		// Low dimensions
		D = 2;

		DifferentiableFunction f = new Rastrigins(D);
		// The rastrigin optimum is at vec(0)

		// Optimization bounds: −5.12 ≤ xi ≤ 5.12
		double [] L = new double[D];
		double [] U = new double[D];
		double [] start = new double[D];
		for(int i=0; i<D; i++) {
			L[i] = -5.12;
			U[i] = 5.12;
			start[i] = Prng.nextDouble();
		}
		f.setPoint(start);
		log.info("starting pt = ("+start[0]+", "+start[1]+")");
		Bounds b = new Bounds(L, U);

		ConstrainedDifferentiableFunction g = new FunctionOpts.DifferentiableFunctionWithConstraints(f, b);
		
		SquaredExpKernel kernel = new SquaredExpKernel();
		GPGO opt = new GPGO(g, kernel, maxiter);

		opt.minimize();

		double [] opt_point = f.getPoint();
		double opt_val = f.getValue();

		log.info("found opt val = " + opt_val);
		assertEquals(0, opt_val, 0.1);

		// see how close we are to the opt point
		for(int i=0; i<D; i++) {
			assertEquals(0, opt_point[i], 0.1);
		}
	}
	
	public static void main(String [] args) {
		GPGOTest tester = new GPGOTest();
		tester.myopicLossTest();
	}
	
}
