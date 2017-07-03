package edu.jhu.hlt.optimize;

import edu.jhu.hlt.optimize.function.DifferentiableFunction;

public class LBFGSTest extends AbstractOptimizerTest {

    protected Optimizer<DifferentiableFunction> getOptimizer() {
        return new LBFGS();
    }
    
    protected boolean supportsL1Regularization() { return false; }
    
}
