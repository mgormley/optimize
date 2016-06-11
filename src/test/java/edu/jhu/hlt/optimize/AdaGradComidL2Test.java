package edu.jhu.hlt.optimize;

import org.junit.Test;

import edu.jhu.hlt.optimize.AdaGradComidL2.AdaGradComidL2Prm;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.functions.SumSquares;
import edu.jhu.hlt.util.JUnitUtils;
import edu.jhu.hlt.util.math.Vectors;
import edu.jhu.prim.vector.IntDoubleDenseVector;

public class AdaGradComidL2Test extends AbstractBatchOptimizerTest {

    @Override
    protected Optimizer<DifferentiableBatchFunction> getOptimizer(String id) {
        AdaGradComidL2Prm prm = getOptimizerPrm();
        return new AdaGradComidL2(prm);
    }

    protected AdaGradComidL2Prm getOptimizerPrm() {
        AdaGradComidL2Prm prm = new AdaGradComidL2Prm();
        prm.eta = 0.1 * 100;
        prm.sched = null;
        //prm.sched.setEta0(0.1 * 10);
        prm.numPasses = 100;
        prm.batchSize = 1;
        prm.autoSelectLr = false;
        prm.l2Lambda = 0.0;
        return prm;
    }

    protected Optimizer<DifferentiableBatchFunction> getRegularizedOptimizer(double l1Lambda, double l2Lambda, String id) {
        AdaGradComidL2Prm prm = getOptimizerPrm();
        if (l1Lambda != 0) { return super.getRegularizedOptimizer(l1Lambda, l2Lambda, id); }
        prm.l2Lambda = l2Lambda;
        return new AdaGradComidL2(prm);
    }
    
    protected double getL1EqualityThreshold() { return 0.4; }
    
    @Test
    public void testAutoSelectLr() {
        {
            // Test with the initial learning rate too small
            runSgdAutoSelectLr(0.05);
        }

        {
            // Test with the initial learning rate too large
            runSgdAutoSelectLr(10);        
        }
    }

    public static void runSgdAutoSelectLr(double eta) {
        AdaGradComidL2Prm prm = new AdaGradComidL2Prm();
        prm.sched = null;
        prm.eta = eta;
        prm.numPasses = 7;
        prm.batchSize = 1;
        prm.autoSelectLr = true;
        AdaGradComidL2 opt = new AdaGradComidL2(prm);
        
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 3, -5, 11};
        opt.maximize(negate(bf(new SumSquares(offsets))), new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(offsets, max, 1e-1);
    }
    
}