package edu.jhu.hlt.optimize;

import edu.jhu.hlt.optimize.AdaGradComidL2.AdaGradComidL2Prm;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;

public class AdaGradComidL2Test extends AbstractBatchOptimizerTest {

    @Override
    protected Optimizer<DifferentiableBatchFunction> getOptimizer() {
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

    protected Optimizer<DifferentiableBatchFunction> getRegularizedOptimizer(double l1Lambda, double l2Lambda) {
        AdaGradComidL2Prm prm = getOptimizerPrm();
        if (l1Lambda != 0) { return super.getRegularizedOptimizer(l1Lambda, l2Lambda); }
        prm.l2Lambda = l2Lambda;
        return new AdaGradComidL2(prm);
    }
    
    protected double getL1EqualityThreshold() { return 0.4; }
    
}