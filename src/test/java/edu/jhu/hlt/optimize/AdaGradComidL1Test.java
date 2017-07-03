package edu.jhu.hlt.optimize;

import org.junit.Test;

import edu.jhu.hlt.optimize.AdaGradComidL1.AdaGradComidL1Prm;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.functions.SumSquares;
import edu.jhu.hlt.util.JUnitUtils;
import edu.jhu.hlt.util.math.Vectors;
import edu.jhu.prim.vector.IntDoubleDenseVector;

public class AdaGradComidL1Test extends AbstractBatchOptimizerTest {

    @Override
    protected Optimizer<DifferentiableBatchFunction> getOptimizer(String id) {
        AdaGradComidL1Prm prm = getOptimizerPrm();
        return new AdaGradComidL1(prm);
    }

    protected AdaGradComidL1Prm getOptimizerPrm() {
        AdaGradComidL1Prm prm = new AdaGradComidL1Prm();
        prm.eta = 0.1 * 100;
        prm.sched = null;
        //prm.sched.setEta0(0.1 * 10);
        prm.numPasses = 100;
        prm.batchSize = 1;
        prm.autoSelectLr = false;
        prm.l1Lambda = 0.0;
        return prm;
    }    

    protected Optimizer<DifferentiableBatchFunction> getRegularizedOptimizer(double l1Lambda, double l2Lambda, String id) {
        AdaGradComidL1Prm prm = getOptimizerPrm();
        prm.l1Lambda = l1Lambda;
        if (l2Lambda != 0) { return super.getRegularizedOptimizer(l1Lambda, l2Lambda, id); }
        return new AdaGradComidL1(prm);
    }
    

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
        AdaGradComidL1Prm prm = new AdaGradComidL1Prm();
        prm.sched = null;
        prm.eta = eta;
        prm.numPasses = 7;
        prm.batchSize = 1;
        prm.autoSelectLr = true;
        AdaGradComidL1 opt = new AdaGradComidL1(prm);
        
        double[] x = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 3, -5, 11};
        opt.minimize(bf(new SumSquares(offsets)), new IntDoubleDenseVector(x));
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(offsets, x, 1e-1);
    }
    
}