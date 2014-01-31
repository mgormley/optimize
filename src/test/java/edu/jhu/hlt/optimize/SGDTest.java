package edu.jhu.hlt.optimize;

import edu.jhu.hlt.optimize.SGD.SGDPrm;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;

public class SGDTest extends AbstractBatchOptimizerTest {

    @Override
    protected Optimizer<DifferentiableBatchFunction> getOptimizer() {
        SGDPrm prm = new SGDPrm();
        prm.initialLr = 0.1 * 10;
        prm.numPasses = 100;
        prm.batchSize = 1;
        return new SGD(prm);
    }

}