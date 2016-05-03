package edu.jhu.hlt.optimize;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import edu.jhu.hlt.optimize.function.BatchFunctionOpts;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.DifferentiableFunctionOpts;
import edu.jhu.hlt.optimize.function.DifferentiableFunctionOpts.NegateFunction;
import edu.jhu.hlt.optimize.function.FunctionOpts;
import edu.jhu.hlt.optimize.functions.SumSquares;
import edu.jhu.hlt.optimize.functions.WeightedSphereModel;
import edu.jhu.hlt.optimize.functions.XSquared;
import edu.jhu.hlt.util.JUnitUtils;
import edu.jhu.hlt.util.math.Vectors;
import edu.jhu.prim.util.random.Prng;
import edu.jhu.prim.vector.IntDoubleDenseVector;

public abstract class AbstractOptimizerTest {

    protected abstract Optimizer<DifferentiableFunction> getOptimizer();

    protected double getL1EqualityThreshold() { return 1e-13; }

    public static DifferentiableFunction negate(DifferentiableFunction f) {
        return new DifferentiableFunctionOpts.NegateFunction(f);
    }

    protected Optimizer<DifferentiableFunction> getRegularizedOptimizer(final double l1Lambda, final double l2Lambda) {
        final Optimizer<DifferentiableFunction> opt = getOptimizer();        
        return DifferentiableFunctionOpts.getRegularizedOptimizer(opt, l1Lambda, l2Lambda);
    }
    
    @Test
    public void testNegXSquared() {
        Optimizer<DifferentiableFunction> opt = getOptimizer();
        double[] max = new double[]{ 9.0 };
        opt.maximize(new NegateFunction(new XSquared()), new IntDoubleDenseVector(max));
        assertEquals(0.0, max[0], 1e-10);      
    }
    
    @Test
    public void testXSquared() {
        Optimizer<DifferentiableFunction> opt = getOptimizer();
        double[] max = new double[]{ 9.0 };
        opt.minimize(new XSquared(), new IntDoubleDenseVector(max));
        assertEquals(0.0, max[0], 1e-10);        
    }
    
    @Test
    public void testNegSumSquares() {
        Optimizer<DifferentiableFunction> opt = getOptimizer();
        double[] initial = new double[3];
        initial[0] = 9;
        initial[1] = 2;
        initial[2] = -7;
        opt.maximize(new NegateFunction(new SumSquares(initial.length)), new IntDoubleDenseVector(initial));
        double[] max = initial;
        JUnitUtils.assertArrayEquals(new double[] {0.0, 0.0, 0.0} , max, 1e-10);
    }
    
    @Test
    public void testOffsetNegSumSquares() {
        Optimizer<DifferentiableFunction> opt = getOptimizer();
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 3, -5, 11};
        opt.maximize(new NegateFunction(new SumSquares(offsets)), new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(offsets, max, 1e-10);
    }

    @Test
    public void testL1RegularizedOffsetNegSumSquaresMax() {
        Optimizer<DifferentiableFunction> opt = getRegularizedOptimizer(1.0, 0.0);
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 0.4, -5, 11};
        double[] expected = new double[]{-0.0, 4.5, -10.5};
        DifferentiableFunction f = negate(new SumSquares(offsets));
//        JUnitUtils.assertArrayEquals(new double[]{0.0, 0.0, 0.0},
//                f.getGradient(new IntDoubleDenseVector(expected)).toNativeArray(),
//                1e-13);
        opt.maximize(negate(new SumSquares(offsets)), new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        assertEquals(expected[0], max[0], getL1EqualityThreshold());
        assertEquals(expected[1], max[1], 1e-10);
        assertEquals(expected[2], max[2], 1e-10);
    }
    
    @Test
    public void testL1RegularizedOffsetNegSumSquaresMin() {
        Optimizer<DifferentiableFunction> opt = getRegularizedOptimizer(1.0, 0.0);
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 0.4, -5, 11};
        double[] expected = new double[]{-0.0, 4.5, -10.5};
        SumSquares f = new SumSquares(offsets);
//        JUnitUtils.assertArrayEquals(new double[]{0.0, 0.0, 0.0},
//                f.getGradient(new IntDoubleDenseVector(expected)).toNativeArray(),
//                1e-13);
        opt.minimize(f, new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        assertEquals(expected[0], max[0], getL1EqualityThreshold());
        assertEquals(expected[1], max[1], 1e-10);
        assertEquals(expected[2], max[2], 1e-10);
    }
    
    @Test
    public void testL2RegularizedOffsetNegSumSquaresMax() {
        Optimizer<DifferentiableFunction> opt = getRegularizedOptimizer(0.0, 1.0);
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 0.4, -5, 11};
        opt.maximize(negate(new SumSquares(offsets)), new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(new double[]{-0.266, 3.333, -7.333}, max, 1e-3);
    }

    @Test
    public void testL2RegularizedOffsetNegSumSquaresMin() {
        Optimizer<DifferentiableFunction> opt = getRegularizedOptimizer(0.0, 1.0);
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 0.4, -5, 11};
        opt.minimize(new SumSquares(offsets), new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(new double[]{-0.266, 3.333, -7.333}, max, 1e-3);
    }

    @Test
    public void testSumSquaresHigherDimension() {
        Optimizer<DifferentiableFunction> opt = getRegularizedOptimizer(0.0, 0.0);
        Prng.seed(12345l);
        int dim = 100;
        double[] initial = new double[dim];
        double[] offsets = new double[dim];
        for (int i=0; i<dim; i++) {
            initial[i] = Prng.nextDouble() * 10 - 5;
            offsets[i] = Prng.nextDouble() * 10 - 5;
        }
        opt.minimize(new SumSquares(offsets), new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(offsets, max, 1e-3);
    }

    @Test
    public void testWeightedSphereModel() {
        Optimizer<DifferentiableFunction> opt = getRegularizedOptimizer(0.0, 0.0);
        Prng.seed(12345l);
        int dim = 100;
        double[] initial = new double[dim];
        for (int i=0; i<dim; i++) {
            initial[i] = Prng.nextDouble() * 10 - 5;
        }
        opt.minimize(new WeightedSphereModel(dim), new IntDoubleDenseVector(initial));
        double[] max = initial;
        double[] global = new double[dim]; // all zeros
        JUnitUtils.assertArrayEquals(global, max, 1e-3);
    }
    
    
}
