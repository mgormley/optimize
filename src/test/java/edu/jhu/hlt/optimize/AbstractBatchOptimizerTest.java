package edu.jhu.hlt.optimize;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import edu.jhu.hlt.optimize.function.BatchFunctionOpts;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.FunctionAsBatchFunction;
import edu.jhu.hlt.optimize.functions.SumSquares;
import edu.jhu.hlt.optimize.functions.XSquared;
import edu.jhu.hlt.util.JUnitUtils;
import edu.jhu.hlt.util.math.Vectors;
import edu.jhu.prim.vector.IntDoubleDenseVector;

public abstract class AbstractBatchOptimizerTest {

    protected abstract Optimizer<DifferentiableBatchFunction> getOptimizer();
    
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
    
    public static DifferentiableBatchFunction bf(DifferentiableFunction f) {
        return new FunctionAsBatchFunction(f, 10);
    }
    
    public static DifferentiableBatchFunction negate(DifferentiableBatchFunction bf) {
        return new BatchFunctionOpts.NegateFunction(bf);
    }
    
}
