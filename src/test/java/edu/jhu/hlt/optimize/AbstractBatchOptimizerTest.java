package edu.jhu.hlt.optimize;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import edu.jhu.hlt.optimize.function.BatchFunctionOpts;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.DifferentiableFunctionOpts;
import edu.jhu.hlt.optimize.function.FunctionAsBatchFunction;
import edu.jhu.hlt.optimize.functions.L1;
import edu.jhu.hlt.optimize.functions.L2;
import edu.jhu.hlt.optimize.functions.SumSquares;
import edu.jhu.hlt.optimize.functions.XSquared;
import edu.jhu.hlt.util.JUnitUtils;
import edu.jhu.hlt.util.math.Vectors;
import edu.jhu.prim.vector.IntDoubleDenseVector;
import edu.jhu.prim.vector.IntDoubleVector;

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
    
    protected Optimizer<DifferentiableBatchFunction> getRegularizedOptimizer(final double l1Lambda, final double l2Lambda) {
        final Optimizer<DifferentiableBatchFunction> opt = getOptimizer();
        
        return new Optimizer<DifferentiableBatchFunction>() {
            
            @Override
            public boolean minimize(DifferentiableBatchFunction function, IntDoubleVector point) {
                return optimize(function, point, false);
            }
            
            @Override
            public boolean maximize(DifferentiableBatchFunction function, IntDoubleVector point) {
                return optimize(function, point, true);
            }
            
            public boolean optimize(DifferentiableBatchFunction objective, IntDoubleVector point, boolean maximize) {
                L1 l1 = new L1(l1Lambda);
                L2 l2 = new L2(1.0 / l2Lambda);
                l1.setNumDimensions(objective.getNumDimensions());
                l2.setNumDimensions(objective.getNumDimensions());
                DifferentiableFunction reg = new DifferentiableFunctionOpts.AddFunctions(l1, l2);

                DifferentiableBatchFunction br = new FunctionAsBatchFunction(reg, objective.getNumExamples());
                DifferentiableBatchFunction nbr = !maximize ? new BatchFunctionOpts.NegateFunction(br) : br;
                DifferentiableBatchFunction fn = new BatchFunctionOpts.AddFunctions(objective, nbr);
                
                if (!maximize) {
                    return opt.minimize(fn, point);   
                } else {
                    return opt.maximize(fn, point);
                }
            }
        };
    }
    
    @Test
    public void testL1RegularizedOffsetNegSumSquaresMax() {
        Optimizer<DifferentiableBatchFunction> opt = getRegularizedOptimizer(1.0, 0.0);
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 0.4, -5, 11};
        opt.maximize(negate(bf(new SumSquares(offsets))), new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(new double[]{-0.0, 4.5, -10.5}, max, 1e-10);
    }
    
    @Test
    public void testL1RegularizedOffsetNegSumSquaresMin() {
        Optimizer<DifferentiableBatchFunction> opt = getRegularizedOptimizer(1.0, 0.0);
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 0.4, -5, 11};
        opt.minimize(bf(new SumSquares(offsets)), new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(new double[]{-0.0, 4.5, -10.5}, max, 1e-10);
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
}
