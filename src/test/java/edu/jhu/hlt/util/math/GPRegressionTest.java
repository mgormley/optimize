package edu.jhu.hlt.util.math;

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
import com.xeiam.xchart.StyleManager.ChartType;
import com.xeiam.xchart.SwingWrapper;

import edu.jhu.hlt.optimize.Function;
import edu.jhu.hlt.optimize.SGDQNCorrectedTest;
import edu.jhu.hlt.optimize.XSquared;
import edu.jhu.hlt.util.math.GPRegression.GPRegressor;
import edu.jhu.hlt.util.math.GPRegression.RegressionResult;

public class GPRegressionTest {

	static Logger log = Logger.getLogger(GPRegressionTest.class);
	
	@Test
	public void regressionTest() {
		
    	BasicConfigurator.configure();
    	Logger.getRootLogger().setLevel(Level.DEBUG);
		
    	// Parameters
    	Kernel kernel = new SquaredExpKernel(1d, 1d);
    	Function f = new XSquared(0);
    	double noise = 0d;
    	
    	// Training data
    	double [][] xs = {{-3},{-2},{-1}, {1}, {2}, {3}};
		RealMatrix X = MatrixUtils.createRealMatrix(xs).transpose();
		double [] ys = new double[xs.length];
		for(int i=0; i<ys.length; i++) {
			ys[i] = f.getValue(xs[i]);
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
