package edu.jhu.hlt.optimize.functions;

import java.awt.Color;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import com.xeiam.xchart.Chart;
import com.xeiam.xchart.ChartBuilder;
import com.xeiam.xchart.Series;
import com.xeiam.xchart.SeriesMarker;
import com.xeiam.xchart.SwingWrapper;
import com.xeiam.xchart.StyleManager.ChartType;

import edu.jhu.hlt.optimize.function.Bounds;
import edu.jhu.hlt.optimize.function.Function;
import edu.jhu.hlt.optimize.function.FunctionOpts;
import edu.jhu.hlt.optimize.function.FunctionOpts.DifferentiableFunctionWithConstraints;
import edu.jhu.hlt.optimize.GradientDescentWithLineSearch;

public class UnevenDecreasingMinimaOptimizedByGD implements Function {
	
	static Logger log = Logger.getLogger(UnevenDecreasingMinimaOptimizedByGD.class);

	Bounds b = Bounds.getUnitBounds(1);
	DifferentiableFunctionWithConstraints f = new FunctionOpts.DifferentiableFunctionWithConstraints(new FunctionOpts.NegateFunction(new UnevenDecreasingMaxima()), b);
	double x;
	
	@Override
	public void setPoint(double[] point) {
		x = point[0];
	}

	@Override
	public double[] getPoint() {
		return new double[] {x};
	}

	@Override
	public double getValue(double[] point) {
		GradientDescentWithLineSearch opt = new GradientDescentWithLineSearch(150);
		opt.minimize(f, point);
		return f.getValue();
	}

	@Override
	public double getValue() {
		return getValue(getPoint());
	}

	@Override
	public int getNumDimensions() {
		return 1;
	}
	
	public static void main(String [] args) {
		
		BasicConfigurator.configure();
		Logger.getRootLogger().setLevel(Level.DEBUG);
		
		UnevenDecreasingMaxima g = new UnevenDecreasingMaxima();
		Function f = new FunctionOpts.NegateFunction(g);
		
		Function fopt = new UnevenDecreasingMinimaOptimizedByGD();
		
		double grid_min = 0.05;
		double grid_max = 0.95;
		double range = grid_max - grid_min;
		int npts = 100;
		double increment = range/(double)npts; 
		
		List<Number> grid = new ArrayList<Number>();
		List<Number> fvals = new ArrayList<Number>();
		List<Number> foptvals = new ArrayList<Number>();
		
		for(double x=grid_min; x<grid_max; x+=increment) {	
			log.info("x="+x);
			double y = f.getValue(new double[] {x});
			double y2 = fopt.getValue(new double[] {x});
			log.info("y="+y);
			log.info("y2="+y2);
			grid.add(x);
			fvals.add(y);
			foptvals.add(y2);
			//System.exit(1);
		}
		
		Chart chart = new ChartBuilder().width(800).height(600).title("ScatterChart04").xAxisTitle("X").yAxisTitle("Y").chartType(ChartType.Line).build();
	
		// Customize Chart
		chart.getStyleManager().setChartTitleVisible(true);
		chart.getStyleManager().setLegendVisible(false);
		chart.getStyleManager().setAxisTitlesVisible(true);
	 
		// Series 0 (observations)
		Series series0 = chart.addSeries("Function", grid, fvals);
		series0.setMarker(SeriesMarker.NONE);
		
		Series series1 = chart.addSeries("Optimized function", grid, foptvals);
		series1.setMarker(SeriesMarker.NONE);
		
		chart.getStyleManager().setYAxisMin(1.0);
		chart.getStyleManager().setYAxisMax(-1.0);
		 
		chart.getStyleManager().setXAxisMin(0.0);
		chart.getStyleManager().setXAxisMax(1.0);
		
	    new SwingWrapper(chart).displayChart();
	}
}