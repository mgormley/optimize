package edu.jhu.hlt.optimize.function;

import edu.jhu.hlt.util.Utilities;

public abstract class AbstractBatchFunction implements BatchFunction {

    @Override
    public abstract int getNumDimensions();
    
    @Override
    public abstract int getNumExamples();

}
