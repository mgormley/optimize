package edu.jhu.hlt.optimize;

import edu.jhu.hlt.optimize.AdaDelta.AdaDeltaPrm;
import edu.jhu.hlt.optimize.AdaGrad.AdaGradPrm;
import edu.jhu.hlt.optimize.SGD.SGDPrm;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;

public class AdaDeltaTest  extends AbstractBatchOptimizerTest {

    @Override
    protected Optimizer<DifferentiableBatchFunction> getOptimizer() {
        AdaDeltaPrm sched = new AdaDeltaPrm();
        sched.decayRate = 0.95;
        sched.constantAddend = Math.pow(Math.E, -6);
        
        SGDPrm prm = new SGDPrm();
        prm.sched = new AdaDelta(sched);
        prm.numPasses = 100;        
        prm.batchSize = 1;
        prm.autoSelectLr = false;
        return new SGD(prm);
    }
    
}