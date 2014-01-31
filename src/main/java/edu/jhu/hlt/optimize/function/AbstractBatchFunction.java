package edu.jhu.hlt.optimize.function;

import edu.jhu.hlt.util.Utilities;
import edu.jhu.prim.vector.IntDoubleVector;

public abstract class AbstractBatchFunction implements BatchFunction {

    @Override
    public double getValue(IntDoubleVector point) {
        return getValue(point, Utilities.getIndexArray(getNumExamples()));
    }

    @Override
    public abstract int getNumDimensions();

    @Override
    public abstract double getValue(IntDoubleVector point, int[] batch);
    
    @Override
    public abstract int getNumExamples();

}
