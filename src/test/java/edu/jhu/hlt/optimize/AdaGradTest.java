package edu.jhu.hlt.optimize;

import edu.jhu.hlt.optimize.AdaGrad.AdaGradPrm;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;

public class AdaGradTest extends AbstractBatchOptimizerTest {

    @Override
    protected Optimizer<DifferentiableBatchFunction> getOptimizer() {
        AdaGradPrm prm = new AdaGradPrm();
        prm.eta = 0.1 * 100;
        prm.sgdPrm.numPasses = 100;        
        prm.sgdPrm.batchSize = 1;
        return new AdaGrad(prm);
    }
    
}
