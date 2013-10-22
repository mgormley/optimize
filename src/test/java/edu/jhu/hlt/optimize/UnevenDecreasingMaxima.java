package edu.jhu.hlt.optimize;

import java.awt.Color;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import com.xeiam.xchart.Chart;
import com.xeiam.xchart.ChartBuilder;
import com.xeiam.xchart.Series;
import com.xeiam.xchart.SeriesLineStyle;
import com.xeiam.xchart.SeriesMarker;
import com.xeiam.xchart.SwingWrapper;
import com.xeiam.xchart.StyleManager.ChartType;

public class UnevenDecreasingMaxima implements DifferentiableFunction {

	static Logger log = Logger.getLogger(UnevenDecreasingMaxima.class);
	
	int n;
	int order = 1; // 1st derivatives only
	double x;
	
	public UnevenDecreasingMaxima() {
		
	}
	
	DerivativeStructure AD_getValue(double [] point) {
		
		DerivativeStructure dx = new DerivativeStructure(1, 1, 0, x);
		
		DerivativeStructure t1 = dx.subtract(0.08).divide(0.854).pow(2).multiply(Math.log(2.0)*-2.0).exp();
		DerivativeStructure t2 = dx.pow(3/4).subtract(0.05).multiply(5.0*Math.PI).sin().pow(6);
		
		return t1.multiply(t2);
	}
	
	@Override
	public void setPoint(double[] point) {
		x = point[0];
	}

	@Override
	public double[] getPoint() {
		return new double [] {x};
	}

	@Override
	public double getValue(double[] point) {
		setPoint(point);
		double tmp1 = -2*Math.log(2.0)*((x-0.08)/0.854)*((x-0.08)/0.854);
	    double tmp2 = Math.sin( 5*Math.PI*(Math.pow(x,3.0/4.0)-0.05) );
	    return Math.exp(tmp1) * Math.pow(tmp2, 6);
	}

	@Override
	public double getValue() {
		return getValue(getPoint());
	}

	@Override
	public int getNumDimensions() {
		return 1;
	}

	@Override
	public void getGradient(double[] gradient) {
		DerivativeStructure value = AD_getValue(getPoint());
		for(int i=0; i<n; i++) {
			int [] orders = new int[n];
			orders[i] = 1;
			gradient[i] = value.getPartialDerivative(orders);
		}
	}
	
	public static void main(String [] args) {
		
		BasicConfigurator.configure();
		Logger.getRootLogger().setLevel(Level.DEBUG);
		
		UnevenDecreasingMaxima g = new UnevenDecreasingMaxima();
		Function f = new FunctionOpts.NegateFunction(g);
		
		double grid_min = 0.0;
		double grid_max = 1.0;
		double range = grid_max - grid_min;
		int npts = 500;
		double increment = range/(double)npts; 
		
		List<Number> grid = new ArrayList<Number>();
		List<Number> fvals = new ArrayList<Number>();
		
		for(double x=grid_min; x<grid_max; x+=increment) {	
			log.info("x="+x);
			double y = f.getValue(new double[] {x});
			log.info("y="+y);
			grid.add(x);
			fvals.add(y);
		}
		
		Chart chart = new ChartBuilder().width(800).height(600).title("ScatterChart04").xAxisTitle("X").yAxisTitle("Y").chartType(ChartType.Line).build();
	
		// Customize Chart
		chart.getStyleManager().setChartTitleVisible(false);
		chart.getStyleManager().setLegendVisible(true);
		chart.getStyleManager().setAxisTitlesVisible(false);
	 
		// Series 0 (observations)
		Series series0 = chart.addSeries("Observations", grid, fvals);
		series0.setMarker(SeriesMarker.NONE);
		series0.setMarkerColor(Color.BLACK);
		
		chart.getStyleManager().setYAxisMin(0);
		chart.getStyleManager().setYAxisMax(1);
		 
		chart.getStyleManager().setXAxisMin(0);
		chart.getStyleManager().setXAxisMax(1);
		
	    new SwingWrapper(chart).displayChart();
	}
}
