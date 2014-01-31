package edu.jhu.hlt.optimize.function;

import edu.jhu.hlt.util.Utilities;

/**
 * @author mgormley
 */
public abstract class AbstractDifferentiableBatchFunction extends AbstractBatchFunction implements DifferentiableBatchFunction {

    @Override
    public abstract int getNumDimensions();
    
    @Override
    public abstract int getNumExamples();

}
