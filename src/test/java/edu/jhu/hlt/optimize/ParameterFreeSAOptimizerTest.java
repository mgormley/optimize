package edu.jhu.hlt.optimize;

import static org.junit.Assert.assertEquals;

import java.awt.Color;
import java.util.ArrayList;
import java.util.List;

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

import edu.jhu.hlt.util.Prng;

public class ParameterFreeSAOptimizerTest {
	
	static Logger log = Logger.getLogger(ParameterFreeSAOptimizerTest.class);
	
	@Test
	public void Rastrigins() {

		BasicConfigurator.configure();
		Logger.getRootLogger().setLevel(Level.DEBUG);

		Prng.seed(42);

		int D;
		int maxiter = 1000;

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
		ParameterFreeSAOptimizer opt = new ParameterFreeSAOptimizer(g, maxiter);

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
		
		BasicConfigurator.configure();
		Logger.getRootLogger().setLevel(Level.DEBUG);
		
		double grid_min = -5;
		double grid_max = +5;
		double range = grid_max - grid_min;
		int npts = 50;
		double increment = range/(double)npts; 
		List<Number> grid = new ArrayList<Number>();
		List<Number> fvals = new ArrayList<Number>();
		for(double x=grid_min; x<grid_max; x+=increment) {
			grid.add(x);
			double avg = 0;
			for(int i=0; i<100; i++) {
				avg += VFSAOptimizer.getCauchy(x);
			}
			avg /= 100;
			log.info("f("+x+")="+avg);
			if(Double.isNaN(avg)) {
				avg = 0;
			}
			fvals.add(avg);
		}
		
		// Create Chart
		Chart chart = new ChartBuilder().width(800).height(600).title("ScatterChart04").xAxisTitle("X").yAxisTitle("Y").chartType(ChartType.Line).build();
				
		// Customize Chart
		chart.getStyleManager().setChartTitleVisible(false);
		chart.getStyleManager().setLegendVisible(true);
		chart.getStyleManager().setAxisTitlesVisible(false);
			 
		// Series 0 (observations)
		Series series0 = chart.addSeries("Observations", grid, fvals);
		series0.setMarker(SeriesMarker.SQUARE);
		series0.setMarkerColor(Color.BLACK);
		
	    new SwingWrapper(chart).displayChart();
	}
}
