package edu.jhu.hlt.optimize;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import edu.jhu.hlt.optimize.BottouSchedule.BottouSchedulePrm;
import edu.jhu.hlt.optimize.SGD.SGDPrm;
import edu.jhu.hlt.optimize.function.AbstractDifferentiableBatchFunction;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.hlt.optimize.functions.SumSquares;
import edu.jhu.hlt.util.JUnitUtils;
import edu.jhu.hlt.util.math.Vectors;
import edu.jhu.prim.util.random.Prng;
import edu.jhu.prim.vector.IntDoubleDenseVector;
import edu.jhu.prim.vector.IntDoubleVector;

public class SGDTest extends AbstractBatchOptimizerTest {

    @Override
    protected Optimizer<DifferentiableBatchFunction> getOptimizer(String id) {
        SGDPrm prm = new SGDPrm();
        prm.sched.setEta0(0.1 * 10);
        prm.numPasses = 100;
        prm.batchSize = 1;
        prm.autoSelectLr = false;
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
        
        double[] x = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 3, -5, 11};
        opt.minimize(bf(new SumSquares(offsets)), new IntDoubleDenseVector(x));
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(offsets, x, 1e-1);
    }
    
    private static class MyFnForAvg extends AbstractDifferentiableBatchFunction implements DifferentiableBatchFunction {

        @Override
        public ValueGradient getValueGradient(IntDoubleVector point, int[] batch) {
            IntDoubleDenseVector g = new IntDoubleDenseVector();
            double val = 0;
            for (int i=0; i<batch.length; i++) {
                int c = (batch[i] % 2 == 0) ? -1 : 1;
                val += c;
                g.set(0, -c);
                g.set(1, c);
            }
            return new ValueGradient(val, g);
        }

        @Override
        public IntDoubleVector getGradient(IntDoubleVector point, int[] batch) {
            return getValueGradient(point, batch).getGradient();
        }

        @Override
        public double getValue(IntDoubleVector point, int[] batch) {
            return getValueGradient(point, batch).getValue();
        }

        @Override
        public int getNumDimensions() {
            return 2;
        }

        @Override
        public int getNumExamples() {
            return 5;
        }
        
    }
    
    @Test
    public void testAveraging() {
        Prng.seed(123456789101112l);
        SGDPrm prm = new SGDPrm();
        prm.sched.setEta0(0.1 * 10);
        prm.numPasses = 1;
        prm.batchSize = 1;
        prm.autoSelectLr = false;
        // Use a schedule which always has learning rate 1.0.
        BottouSchedulePrm sPrm = new BottouSchedulePrm();
        sPrm.initialLr = 1;
        sPrm.lambda = 0;
        sPrm.power = 1;
        prm.sched = new BottouSchedule(sPrm);

        prm.averaging = false;
        {
            SGD sgd = new SGD(prm);
            IntDoubleDenseVector point = new IntDoubleDenseVector(2);
            sgd.minimize(new MyFnForAvg(), point);
            System.out.println(point);
            assertEquals(-1.0, point.get(0), 1e-13);
            assertEquals(1.0, point.get(1), 1e-13);
        }
        
        prm.passToStartAvg = 0;
        prm.averaging = true;
        {
            SGD sgd = new SGD(prm);
            IntDoubleDenseVector point = new IntDoubleDenseVector(2);
            sgd.minimize(new MyFnForAvg(), point);
            System.out.println(point);
            assertEquals((1 + 0 + 1 + 0 + 1) / 5.0, point.get(0), 1e-13);
            assertEquals((-1 + -2 + -1 + 0 + 1) / 5.0, point.get(1), 1e-13);
        }
        
    }
    
    // Below, L1 regularization with SGD doesn't land at the same spot as SGDFobos.
    
    @Test
    public void testL1RegularizedOffsetSumSquares() {
        Optimizer<DifferentiableBatchFunction> opt = getRegularizedOptimizer(1.0, 0.0, null);
        double[] x = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 0.4, -5, 11};
        double[] expected = new double[]{-0.00130530530530, 4.5, -10.5};
        DifferentiableBatchFunction f = bf(new SumSquares(offsets));
        JUnitUtils.assertArrayEquals(new double[]{0.797, -1.0, 1.0},
                f.getGradient(new IntDoubleDenseVector(expected)).toNativeArray(),
                1e-3);
        opt.minimize(f, new IntDoubleDenseVector(x));
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(expected, x, 1e-10);
    }
    
}