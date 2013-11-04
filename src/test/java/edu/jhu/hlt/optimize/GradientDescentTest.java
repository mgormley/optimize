package edu.jhu.hlt.optimize;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import edu.jhu.hlt.optimize.function.FunctionOpts;
import edu.jhu.hlt.optimize.functions.SumSquares;
import edu.jhu.hlt.optimize.functions.XSquared;
import edu.jhu.hlt.util.JUnitUtils;
import edu.jhu.hlt.util.Utilities;
import edu.jhu.hlt.util.math.Vectors;

public class GradientDescentTest {

    @Test
    public void testNegXSquared() {
        GradientDescent opt = new GradientDescent(0.1, 100);
        double[] max = new double[]{ 9.0 };
        opt.maximize(new FunctionOpts.NegateFunction(new XSquared()), max);
        assertEquals(0.0, max[0], 1e-10);      
    }
    
    @Test
    public void testXSquared() {
        GradientDescent opt = new GradientDescent(0.1, 100);
        double[] max = new double[]{ 9.0 };
        opt.minimize(new XSquared(), max);
        assertEquals(0.0, max[0], 1e-10);        
    }
    
    @Test
    public void testNegSumSquares() {
        GradientDescent opt = new GradientDescent(0.1, 100);
        double[] initial = new double[3];
        initial[0] = 9;
        initial[1] = 2;
        initial[2] = -7;
        opt.maximize(new FunctionOpts.NegateFunction(new SumSquares(initial.length)), initial);
        double[] max = initial;
        JUnitUtils.assertArrayEquals(new double[] {0.0, 0.0, 0.0} , max, 1e-10);
    }
    
    @Test
    public void testOffsetNegSumSquares() {
        GradientDescent opt = new GradientDescent(0.1, 100);
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 3, -5, 11};
        opt.maximize(new FunctionOpts.NegateFunction(new SumSquares(offsets)), initial);
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(offsets, max, 1e-10);
    }
}
