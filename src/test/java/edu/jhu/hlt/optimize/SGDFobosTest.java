package edu.jhu.hlt.optimize;

import edu.jhu.hlt.optimize.SGDFobos.SGDFobosPrm;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;

public class SGDFobosTest extends AbstractBatchOptimizerTest {

    @Override
    protected Optimizer<DifferentiableBatchFunction> getOptimizer(String id) {
        SGDFobosPrm prm = getOptimizerPrm();
        return new SGDFobos(prm);
    }

    protected SGDFobosPrm getOptimizerPrm() {
        SGDFobosPrm prm = new SGDFobosPrm();
        prm.sched.setEta0(0.1 * 10);
        prm.numPasses = 100;
        prm.batchSize = 1;
        prm.autoSelectLr = false;
        prm.l1Lambda = 0.0;
        return prm;
    }    

    protected Optimizer<DifferentiableBatchFunction> getRegularizedOptimizer(double l1Lambda, double l2Lambda, String id) {
        SGDFobosPrm prm = getOptimizerPrm();
        prm.l1Lambda = l1Lambda;
        prm.l2Lambda = l2Lambda;
        return new SGDFobos(prm);
    }

}