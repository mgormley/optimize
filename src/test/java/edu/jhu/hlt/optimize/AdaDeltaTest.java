package edu.jhu.hlt.optimize;

import edu.jhu.hlt.optimize.AdaDelta.AdaDeltaPrm;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;

public class AdaDeltaTest  extends AbstractBatchOptimizerTest {

    @Override
    protected Optimizer<DifferentiableBatchFunction> getOptimizer() {
        AdaDeltaPrm prm = new AdaDeltaPrm();
        prm.decayRate = 0.95;
        prm.constantAddend = Math.pow(Math.E, -6);
        prm.sgdPrm.numPasses = 100;        
        prm.sgdPrm.batchSize = 1;
        return new AdaDelta(prm);
    }
    
}