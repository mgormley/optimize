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
import com.xeiam.xchart.SeriesMarker;
import com.xeiam.xchart.SwingWrapper;
import com.xeiam.xchart.StyleManager.ChartType;

import edu.jhu.hlt.optimize.GPGO.ExpectedMyopicLoss;
import edu.jhu.hlt.util.math.Kernel;
import edu.jhu.hlt.util.math.SquaredExpKernel;
import edu.jhu.hlt.util.math.GPRegression.GPRegressor;
import edu.jhu.hlt.util.math.GPRegression.RegressionResult;

public class GPGOTest {

	static Logger log = Logger.getLogger(GPGOTest.class);
	
	@Test
	public void myopicLossTest() {
		
    	BasicConfigurator.configure();
    	Logger.getRootLogger().setLevel(Level.DEBUG);
		
    	// Parameters
    	Kernel kernel = new SquaredExpKernel(1d, 1d);
    	Function f = new XSquared(0);
    	double noise = 0d;
    	
    	// Training data:
    	//   X	inputs
    	//   y  ouputs
    	double [][] xs = {{-3},{-2},{-1}, {1}, {2}, {3}};
		RealMatrix X = MatrixUtils.createRealMatrix(xs).transpose();
		double [] ys = new double[xs.length];
		for(int i=0; i<ys.length; i++) {
			ys[i] = f.getValue(xs[i]);
		}
		RealVector y = new ArrayRealVector(ys);

		// Initialize the GPGO instance
		GPGO opt = new GPGO(f, kernel, X, y, 0d);
		opt.estimatePosterior();
		ExpectedMyopicLoss loss = opt.getExpectedLoss();
		
		double [][] xs2 = {{-4},{-2.5},{-1.5}, {-0.5}, {0.5}, {1.5}, {2.5}, {4}};
		RealMatrix X2 = MatrixUtils.createRealMatrix(xs2).transpose();
		List<Number> xData = new ArrayList<Number>();
		List<Number> yData1 = new ArrayList<Number>();
		List<Number> yData2 = new ArrayList<Number>();
		List<Number> errorBars1 = new ArrayList<Number>();
		for (int i=0; i<X2.getColumnDimension(); i++) {
		  xData.add(xs2[i][0]);
		  RegressionResult pred = opt.getRegressor().predict(X2.getColumnVector(i));
		  double obj = loss.getValue(xs2[i]);
		  
		  // GP predictions
		  yData1.add(pred.mean);
		  errorBars1.add(pred.var);
		  
		  // loss
		  yData2.add(obj);
		}
		
		// Create Chart
		Chart chart = new ChartBuilder().width(800).height(600).title("ScatterChart04").xAxisTitle("X").yAxisTitle("Y").chartType(ChartType.Scatter).build();
		 
		// Customize Chart
		chart.getStyleManager().setChartTitleVisible(false);
		chart.getStyleManager().setLegendVisible(false);
		chart.getStyleManager().setAxisTitlesVisible(false);
		 
		// Series 1 (GP posterior)
		Series series1 = chart.addSeries("Observations", xData, yData1, errorBars1);
		series1.setMarkerColor(Color.RED);
		series1.setMarker(SeriesMarker.SQUARE);
		
		// Series 2 (expected loss)
		Series series2 = chart.addSeries("Predictions", xData, yData2);
		series2.setMarkerColor(Color.GREEN);
		series2.setMarker(SeriesMarker.DIAMOND);
		
		// Show it
	    new SwingWrapper(chart).displayChart();
	    
	    log.info("done!");
		
	}
	
}
