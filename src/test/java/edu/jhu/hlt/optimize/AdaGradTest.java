package edu.jhu.hlt.optimize;

import org.junit.Test;

import edu.jhu.hlt.optimize.AdaGrad.AdaGradPrm;
import edu.jhu.hlt.optimize.SGD.SGDPrm;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;

public class AdaGradTest extends AbstractBatchOptimizerTest {

    @Override
    protected Optimizer<DifferentiableBatchFunction> getOptimizer() {
        AdaGradPrm sched = new AdaGradPrm();
        sched.eta = 0.1 * 100;
        
        SGDPrm prm = new SGDPrm();
        prm.sched = new AdaGrad(sched);
        prm.numPasses = 100;        
        prm.batchSize = 1;
        prm.autoSelectLr = false;
        return new SGD(prm);
    }
    
    @Test
    public void testSgdAutoSelectLr() {
        {
            // Test with the initial learning rate too small
            AdaGradPrm sched = new AdaGradPrm();
            sched.eta = 0.5;
            SGDTest.runSgdAutoSelectLr(new AdaGrad(sched));        
        }
        {
            // Test with the initial learning rate too large
            AdaGradPrm sched = new AdaGradPrm();
            sched.eta = 10;
            SGDTest.runSgdAutoSelectLr(new AdaGrad(sched));  
        }
    }
    
}
