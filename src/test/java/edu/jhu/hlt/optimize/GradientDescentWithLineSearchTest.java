package edu.jhu.hlt.optimize;

import edu.jhu.hlt.optimize.function.DifferentiableFunction;

public class GradientDescentWithLineSearchTest extends AbstractOptimizerTest {

    @Override
    protected Optimizer<DifferentiableFunction> getOptimizer() {
        return new GradientDescentWithLineSearch(100);
    }
    
}