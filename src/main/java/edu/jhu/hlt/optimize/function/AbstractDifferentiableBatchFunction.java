package edu.jhu.hlt.optimize.function;

import edu.jhu.hlt.util.Utilities;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * @author mgormley
 */
public abstract class AbstractDifferentiableBatchFunction extends AbstractBatchFunction implements DifferentiableBatchFunction {

    public ValueGradient getValueGradient(IntDoubleVector point) {
        return getValueGradient(point, Utilities.getIndexArray(getNumExamples()));
    }
    
    @Override
    public IntDoubleVector getGradient(IntDoubleVector point) {
        return getGradient(point, Utilities.getIndexArray(getNumExamples()));
    }
    
    @Override
    public abstract ValueGradient getValueGradient(IntDoubleVector point, int[] batch);

    @Override
    public abstract IntDoubleVector getGradient(IntDoubleVector point, int[] batch);

    @Override
    public abstract int getNumDimensions();

    @Override
    public abstract double getValue(IntDoubleVector point, int[] batch);
    
    @Override
    public abstract int getNumExamples();

}
