package edu.jhu.hlt.optimize;

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
    	Kernel kernel = new SquaredExpKernel(5, 0.1);
    	
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
		A[0] = -10.0;
		B[0] = +10.0;
		Bounds bounds = new Bounds(A, B);
		GPGO opt = new GPGO(f, kernel, bounds, X, y, 0d);
		
		// Estimate the GP posterior
		opt.estimatePosterior();
		
		// Try optimizing the expected loss given this posterior
		ExpectedMyopicLoss loss = opt.getExpectedLoss();
		VFSAOptimizer sa = new VFSAOptimizer(loss, bounds);
		sa.minimize();
		double xguess = loss.getPoint()[0];
		double yguess = loss.getValue();
		List<Number> next_x = new ArrayList<Number>();
		next_x.add(xguess);
		List<Number> next_y = new ArrayList<Number>();
		next_y.add(yguess);
		
		log.info("guess input = " + loss.getPoint()[0]);
		log.info("guess output = " + loss.getValue());
		
		List<Number> posterior_mean = new ArrayList<Number>();
		List<Number> posterior_var = new ArrayList<Number>();
		List<Number> eloss = new ArrayList<Number>();
		
		double grid_min = -2;
		double grid_max = 2;
		double range = grid_max - grid_min;
		int npts = 50;
		double increment = range/(double)npts; 
		List<Number> grid = new ArrayList<Number>();
		
		// Series 3 (actual function)
		List<Number> fvals = new ArrayList<Number>();
		for(double x=grid_min; x<grid_max; x+=increment) {
			log.info("x = " + x);
			
			grid.add(x);
			fvals.add(f.getValue(new double[] {x}));
			RegressionResult pred = opt.getRegressor().predict(new ArrayRealVector(new double[] {x}));
			double obj = opt.getExpectedLoss().computeExpectedLoss(new ArrayRealVector(new double[] {x}));
			//double obj = 0;
			log.info("loss("+x+")="+obj);
			  
			// GP predictions
			posterior_mean.add(pred.mean);
			posterior_var.add(pred.var);
			  
			// Loss
			eloss.add(obj);
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
		
		//Series series4 = chart.addSeries("Next eval", next_x, next_y);
		//series4.setMarker(SeriesMarker.TRIANGLE_DOWN);
		//series4.setMarkerColor(Color.CYAN);
		
		// Show it
	    new SwingWrapper(chart).displayChart();
	    
	    log.info("done!");
		
	}
	
	public static void main(String [] args) {
		GPGOTest tester = new GPGOTest();
		tester.myopicLossTest();
	}
	
}
