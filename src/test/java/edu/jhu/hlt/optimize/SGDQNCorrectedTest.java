package edu.jhu.hlt.optimize;

import static org.junit.Assert.assertEquals;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.junit.Test;

import edu.jhu.hlt.optimize.GradientDescentTest.SumSquares;
import edu.jhu.hlt.optimize.GradientDescentTest.XSquared;
import edu.jhu.hlt.util.JUnitUtils;
import edu.jhu.hlt.util.math.Vectors;

public class SGDQNCorrectedTest {

	static Logger log = Logger.getLogger(SGDQNCorrectedTest.class);
	
    @Test
    public void testNegXSquared() {
    	BasicConfigurator.configure();
    	Logger.getRootLogger().setLevel(Level.DEBUG);
        double[] max = new double[]{ 9.0 };
        DifferentiableFunction f = negate(new XSquared());
        f.setPoint(max);
        SGDQNCorrected opt = getNewSgdQN(f, 100);
        opt.maximize();
        assertEquals(0.0, max[0], 1e-10);
    }
    
    @Test
    public void testXSquared() {
    	BasicConfigurator.configure();
    	Logger.getRootLogger().setLevel(Level.DEBUG);
        DifferentiableFunction f = new XSquared();
        f.setPoint( new double[] {9.0} );
        SGDQNCorrected opt = getNewSgdQN(f, 100);
        opt.minimize();
        double [] result = f.getPoint();
        assertEquals(0.0, result[0], 1e-10);
    }
    
    @Test
    public void testNegSumSquares() {
    	BasicConfigurator.configure();
    	Logger.getRootLogger().setLevel(Level.DEBUG);
        double[] initial = new double[3];
        initial[0] = 9;
        initial[1] = 2;
        initial[2] = -7;
        DifferentiableFunction f = negate(new SumSquares(initial.length));
        f.setPoint(initial);
        SGDQNCorrected opt = getNewSgdQN(f, 100);
        opt.maximize();
        double[] max = initial;
        JUnitUtils.assertArrayEquals(new double[] {0.0, 0.0, 0.0} , max, 1e-10);
    }
    
    @Test
    public void testOffsetNegSumSquares() {
    	BasicConfigurator.configure();
    	Logger.getRootLogger().setLevel(Level.DEBUG);
    	double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 3, -5, 11};
    	DifferentiableFunction f = negate(new SumSquares(offsets));
    	f.setPoint(initial);
        SGDQNCorrected opt = getNewSgdQN(f, 100);
        opt.maximize();
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(offsets, max, 1e-10);
    }
    
    public static SGDQNCorrected getNewSgdQN(DifferentiableFunction f, int T) {
        return new SGDQNCorrected(f, T);
    }
    
    public static DifferentiableFunction negate(DifferentiableFunction f) {
        return new FunctionOpts.NegateFunction(f);
    }
}
