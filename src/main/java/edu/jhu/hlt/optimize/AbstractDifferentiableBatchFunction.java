package edu.jhu.hlt.optimize;

import edu.jhu.hlt.util.Utilities;

public abstract class AbstractDifferentiableBatchFunction extends AbstractBatchFunction implements DifferentiableBatchFunction {

    @Override
    public void getGradient(double[] gradient) {
        getGradient(Utilities.getIndexArray(getNumExamples()), gradient);
    }

    @Override
    public abstract void getGradient(int[] batch, double[] gradient);

    @Override
    public abstract int getNumDimensions();

    @Override
    public abstract void setPoint(double[] point);

    @Override
    public abstract double getValue(int[] batch);
    
    @Override
    public abstract int getNumExamples();

}
