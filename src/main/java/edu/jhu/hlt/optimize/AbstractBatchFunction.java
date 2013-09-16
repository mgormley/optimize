package edu.jhu.hlt.optimize;

import edu.jhu.hlt.util.Utilities;

public abstract class AbstractBatchFunction implements BatchFunction {

    @Override
    public double getValue() {
        return getValue(Utilities.getIndexArray(getNumExamples()));
    }

    @Override
    public abstract int getNumDimensions();

    @Override
    public abstract void setPoint(double[] point);

    @Override
    public abstract double getValue(int[] batch);
    
    @Override
    public abstract int getNumExamples();

}
