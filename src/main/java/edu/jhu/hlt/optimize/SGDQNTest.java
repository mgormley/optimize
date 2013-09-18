package edu.jhu.hlt.optimize;

import static org.junit.Assert.assertEquals;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.junit.Test;

import edu.jhu.hlt.optimize.GradientDescentTest.SumSquares;
import edu.jhu.hlt.optimize.GradientDescentTest.XSquared;
import edu.jhu.hlt.util.JUnitUtils;
import edu.jhu.hlt.util.math.Vectors;

public class SGDQNTest {

    @Test
    public void testNegXSquared() {
    	Logger.getRootLogger().setLevel(Level.DEBUG);
        double[] max = new double[]{ 9.0 };
        DifferentiableFunction f = negate(new XSquared());
        f.setPoint(max);
        SGDQN opt = getNewSgdQN(f, 100);
        opt.maximize();
        assertEquals(0.0, max[0], 1e-10);
    }
    
    @Test
    public void testXSquared() {
    	Logger.getRootLogger().setLevel(Level.DEBUG);
        double[] max = new double[]{ 9.0 };
        DifferentiableFunction f = new XSquared();
        f.setPoint(max);
        SGDQN opt = getNewSgdQN(f, 100);
        opt.minimize();
        assertEquals(0.0, max[0], 1e-10);
    }
    
    @Test
    public void testNegSumSquares() {
    	Logger.getRootLogger().setLevel(Level.DEBUG);
        double[] initial = new double[3];
        initial[0] = 9;
        initial[1] = 2;
        initial[2] = -7;
        DifferentiableFunction f = negate(new SumSquares(initial.length));
        f.setPoint(initial);
        SGDQN opt = getNewSgdQN(f, 100);
        opt.maximize();
        double[] max = initial;
        JUnitUtils.assertArrayEquals(new double[] {0.0, 0.0, 0.0} , max, 1e-10);
    }
    
    @Test
    public void testOffsetNegSumSquares() {
    	Logger.getRootLogger().setLevel(Level.DEBUG);
    	double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 3, -5, 11};
    	DifferentiableFunction f = negate(new SumSquares(offsets));
    	f.setPoint(initial);
        SGDQN opt = getNewSgdQN(f, 100);
        opt.maximize();
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(offsets, max, 1e-10);
    }
    
    public static SGDQN getNewSgdQN(DifferentiableFunction f, int T) {
        return new SGDQN(f, T);
    }
    
    public static DifferentiableFunction negate(DifferentiableFunction f) {
        return new FunctionOpts.NegateFunction(f);
    }
}
