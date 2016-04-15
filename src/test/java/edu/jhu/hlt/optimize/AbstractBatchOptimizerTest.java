package edu.jhu.hlt.optimize;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import edu.jhu.hlt.optimize.function.BatchFunctionOpts;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.FunctionAsBatchFunction;
import edu.jhu.hlt.optimize.functions.SumSquares;
import edu.jhu.hlt.optimize.functions.WeightedSphereModel;
import edu.jhu.hlt.optimize.functions.XSquared;
import edu.jhu.hlt.util.JUnitUtils;
import edu.jhu.hlt.util.math.Vectors;
import edu.jhu.prim.util.random.Prng;
import edu.jhu.prim.vector.IntDoubleDenseVector;

/**
 * Ideas for functions to optimize: 
 * 1. http://www.robertmarks.org/Classes/ENGR5358/Papers/functions.pdf
 * 2. http://arxiv.org/pdf/1308.4008.pdf
 * @author mgormley
 */
public abstract class AbstractBatchOptimizerTest {

    protected abstract Optimizer<DifferentiableBatchFunction> getOptimizer();

    protected double getL1EqualityThreshold() { return 1e-13; }
    
    public static DifferentiableBatchFunction bf(DifferentiableFunction f) {
        return new FunctionAsBatchFunction(f, 10);
    }
    
    public static DifferentiableBatchFunction negate(DifferentiableBatchFunction bf) {
        return new BatchFunctionOpts.NegateFunction(bf);
    }
    
    protected Optimizer<DifferentiableBatchFunction> getRegularizedOptimizer(final double l1Lambda, final double l2Lambda) {
        final Optimizer<DifferentiableBatchFunction> opt = getOptimizer();
        return BatchFunctionOpts.getRegularizedOptimizer(opt, l1Lambda, l2Lambda);
    }
    
    // TODO: Implement a test with a real batch function.
    //    @Test
    //    public void testActualBatchFunction() {
    //        fail("Not yet implemented");
    //    }
    
    @Test
    public void testNegXSquared() {
        Optimizer<DifferentiableBatchFunction> opt = getOptimizer();
        double[] max = new double[]{ 9.0 };
        opt.maximize(negate(bf(new XSquared())), new IntDoubleDenseVector(max));
        assertEquals(0.0, max[0], 1e-10);      
    }
    
    @Test
    public void testXSquared() {
        Optimizer<DifferentiableBatchFunction> opt = getOptimizer();
        double[] max = new double[]{ 9.0 };
        opt.minimize(bf(new XSquared()), new IntDoubleDenseVector(max));
        assertEquals(0.0, max[0], 1e-10);        
    }
    
    @Test
    public void testNegSumSquares() {
        Optimizer<DifferentiableBatchFunction> opt = getOptimizer();
        double[] initial = new double[3];
        initial[0] = 9;
        initial[1] = 2;
        initial[2] = -7;
        opt.maximize(negate(bf(new SumSquares(initial.length))), new IntDoubleDenseVector(initial));
        double[] max = initial;
        JUnitUtils.assertArrayEquals(new double[] {0.0, 0.0, 0.0} , max, 1e-10);
    }
    
    @Test
    public void testOffsetNegSumSquares() {
        Optimizer<DifferentiableBatchFunction> opt = getOptimizer();
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 3, -5, 11};
        opt.maximize(negate(bf(new SumSquares(offsets))), new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(offsets, max, 1e-10);
    }

    @Test
    public void testL1RegularizedOffsetNegSumSquaresMax() {
        Optimizer<DifferentiableBatchFunction> opt = getRegularizedOptimizer(1.0, 0.0);
        double[] initial = new double[] { 0,0,0}; // (different starting point than the Min test below)
        double[] offsets = new double[] { 0.4, -5, 11};
        opt.maximize(negate(bf(new SumSquares(offsets))), new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        assertEquals(-0.0, max[0], getL1EqualityThreshold());
        assertEquals(4.5, max[1], 1e-10);
        assertEquals(-10.5, max[2], 1e-10);
    }
    
    @Test
    public void testL1RegularizedOffsetNegSumSquaresMin() {
        Optimizer<DifferentiableBatchFunction> opt = getRegularizedOptimizer(1.0, 0.0);
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 0.4, -5, 11};
        opt.minimize(bf(new SumSquares(offsets)), new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        assertEquals(-0.0, max[0], getL1EqualityThreshold());
        assertEquals(4.5, max[1], 1e-10);
        assertEquals(-10.5, max[2], 1e-10);
    }
    
    @Test
    public void testL2RegularizedOffsetNegSumSquaresMax() {
        Optimizer<DifferentiableBatchFunction> opt = getRegularizedOptimizer(0.0, 1.0);
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 0.4, -5, 11};
        opt.maximize(negate(bf(new SumSquares(offsets))), new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(new double[]{-0.266, 3.333, -7.333}, max, 1e-3);
    }

    @Test
    public void testL2RegularizedOffsetNegSumSquaresMin() {
        Optimizer<DifferentiableBatchFunction> opt = getRegularizedOptimizer(0.0, 1.0);
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 0.4, -5, 11};
        opt.minimize(bf(new SumSquares(offsets)), new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(new double[]{-0.266, 3.333, -7.333}, max, 1e-3);
    }

    @Test
    public void testSumSquaresHigherDimension() {
        Optimizer<DifferentiableBatchFunction> opt = getRegularizedOptimizer(0.0, 0.0);
        Prng.seed(12345l);
        int dim = 100;
        double[] initial = new double[dim];
        double[] offsets = new double[dim];
        for (int i=0; i<dim; i++) {
            initial[i] = Prng.nextDouble() * 10 - 5;
            offsets[i] = Prng.nextDouble() * 10 - 5;
        }
        opt.minimize(bf(new SumSquares(offsets)), new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(offsets, max, 1e-3);
    }

    @Test
    public void testWeightedSphereModel() {
        Optimizer<DifferentiableBatchFunction> opt = getRegularizedOptimizer(0.0, 0.0);
        Prng.seed(12345l);
        int dim = 100;
        double[] initial = new double[dim];
        for (int i=0; i<dim; i++) {
            initial[i] = Prng.nextDouble() * 10 - 5;
        }
        opt.minimize(bf(new WeightedSphereModel(dim)), new IntDoubleDenseVector(initial));
        double[] max = initial;
        double[] global = new double[dim]; // all zeros
        JUnitUtils.assertArrayEquals(global, max, 1e-3);
    }
    
}
