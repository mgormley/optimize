package edu.jhu.hlt.optimize;

import org.junit.Test;

import edu.jhu.hlt.optimize.AdaGradSchedule.AdaGradSchedulePrm;
import edu.jhu.hlt.optimize.SGD.SGDPrm;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;

public class AdaGradScheduleTest extends AbstractBatchOptimizerTest {

    @Override
    protected Optimizer<DifferentiableBatchFunction> getOptimizer(String id) {
        AdaGradSchedulePrm sched = new AdaGradSchedulePrm();
        sched.eta = 0.1 * 100;
        
        SGDPrm prm = new SGDPrm();
        prm.sched = new AdaGradSchedule(sched);
        prm.numPasses = 100;        
        prm.batchSize = 1;
        prm.autoSelectLr = false;
        return new SGD(prm);
    }
    
    @Test
    public void testSgdAutoSelectLr() {
        {
            // Test with the initial learning rate too small
            AdaGradSchedulePrm sched = new AdaGradSchedulePrm();
            sched.eta = 0.5;
            SGDTest.runSgdAutoSelectLr(new AdaGradSchedule(sched));        
        }
        {
            // Test with the initial learning rate too large
            AdaGradSchedulePrm sched = new AdaGradSchedulePrm();
            sched.eta = 10;
            SGDTest.runSgdAutoSelectLr(new AdaGradSchedule(sched));  
        }
    }

    protected double getL1EqualityThreshold() { return 0.4; }

}
