package edu.jhu.hlt.optimize;

import edu.jhu.hlt.optimize.function.DifferentiableFunction;

public class GradientDescentTest extends AbstractOptimizerTest {

    @Override
    protected Optimizer<DifferentiableFunction> getOptimizer() {
        return new GradientDescent(0.1, 100);
    }
    
}
