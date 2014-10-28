package edu.jhu.hlt.optimize;

import org.junit.Test;

import edu.jhu.hlt.optimize.BottouSchedule.BottouSchedulePrm;
import edu.jhu.hlt.optimize.SGD.SGDPrm;
import edu.jhu.hlt.optimize.function.BatchFunctionOpts;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.DifferentiableFunctionOpts;
import edu.jhu.hlt.optimize.function.FunctionAsBatchFunction;
import edu.jhu.hlt.optimize.function.Regularizer;
import edu.jhu.hlt.optimize.functions.L1;
import edu.jhu.hlt.optimize.functions.L2;
import edu.jhu.hlt.optimize.functions.SumSquares;
import edu.jhu.hlt.util.JUnitUtils;
import edu.jhu.hlt.util.math.Vectors;
import edu.jhu.prim.vector.IntDoubleDenseVector;
import edu.jhu.prim.vector.IntDoubleVector;

/** Test of Averaged Stochastic Gradient Descent. */
public class ASGDTest extends AbstractBatchOptimizerTest {

    @Override
    protected Optimizer<DifferentiableBatchFunction> getOptimizer() {
        BottouSchedulePrm schPrm = new BottouSchedulePrm();
        //schPrm.initialLr = 0.1 * 10;
        schPrm.initialLr = 0.1 * 10;
        schPrm.power = 1.0; // TODO: Setting to 0.75 doesn't work as well on the toy data.
        BottouSchedule sched = new BottouSchedule(schPrm );
        SGDPrm prm = new SGDPrm();
        prm.sched = sched;
        prm.numPasses = 100;
        prm.batchSize = 1;
        prm.autoSelectLr = false;
        prm.averaging = true;
        return new SGD(prm);
    }
    
    @Test
    public void testSgdAutoSelectLr() {
        {
            // Test with the initial learning rate too small
            BottouSchedulePrm sched = new BottouSchedulePrm();
            sched.initialLr = 0.005;
            sched.lambda = 0.1;
            runSgdAutoSelectLr(new BottouSchedule(sched));        
        }

        {
            // Test with the initial learning rate too large
            BottouSchedulePrm sched = new BottouSchedulePrm();
            sched.initialLr = 10;
            sched.lambda = 0.01;
            runSgdAutoSelectLr(new BottouSchedule(sched));        
        }
    }

    public static void runSgdAutoSelectLr(GainSchedule sched) {
        SGDPrm prm = new SGDPrm();
        prm.sched = sched;
        prm.numPasses = 7;
        prm.batchSize = 1;
        prm.autoSelectLr = true;
        SGD opt = new SGD(prm);
        
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 3, -5, 11};
        opt.maximize(negate(bf(new SumSquares(offsets))), new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(offsets, max, 1e-1);
    }
    
    // Below, L1 regularization with SGD doesn't land at the same spot as SGDFobos.
    
    @Test
    public void testL1RegularizedOffsetNegSumSquaresMax() {
        Optimizer<DifferentiableBatchFunction> opt = getRegularizedOptimizer(1.0, 0.0);
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 0.4, -5, 11};
        double[] expected = new double[]{-0.003632323232, 4.5, -10.5};
        DifferentiableBatchFunction f = negate(bf(new SumSquares(offsets)));
        JUnitUtils.assertArrayEquals(new double[]{-0.792, 1.0, -1.0},
                f.getGradient(new IntDoubleDenseVector(expected)).toNativeArray(),
                1e-3);
        opt.maximize(f, new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(expected, max, 1e-10);
    }
    
    @Test
    public void testL1RegularizedOffsetNegSumSquaresMin() {
        Optimizer<DifferentiableBatchFunction> opt = getRegularizedOptimizer(1.0, 0.0);
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 0.4, -5, 11};
        double[] expected = new double[]{-0.003632323232, 4.5, -10.5};
        DifferentiableBatchFunction f = bf(new SumSquares(offsets));
        JUnitUtils.assertArrayEquals(new double[]{0.792, -1.0, 1.0},
                f.getGradient(new IntDoubleDenseVector(expected)).toNativeArray(),
                1e-3);
        opt.minimize(f, new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(expected, max, 1e-10);
    }
    
}