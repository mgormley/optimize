package edu.jhu.hlt.optimize;

import org.junit.Test;

import edu.jhu.hlt.optimize.BottouSchedule.BottouSchedulePrm;
import edu.jhu.hlt.optimize.SGD.SGDPrm;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.functions.SumSquares;
import edu.jhu.hlt.util.JUnitUtils;
import edu.jhu.hlt.util.math.Vectors;
import edu.jhu.prim.vector.IntDoubleDenseVector;

/** Test of Averaged Stochastic Gradient Descent. */
public class ASGDTest extends AbstractBatchOptimizerTest {

    protected double getL1EqualityThreshold() { return 1e-2; }

    @Override
    protected Optimizer<DifferentiableBatchFunction> getOptimizer(String id) {
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
        prm.passToStartAvg = 50;
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
        
}