package edu.jhu.hlt.optimize.function;

import edu.jhu.prim.vector.IntDoubleVector;

public class FunctionAsBatchFunction extends AbstractDifferentiableBatchFunction implements DifferentiableBatchFunction {

    private DifferentiableFunction fn;
    private int numExamples;
    
    public FunctionAsBatchFunction(DifferentiableFunction fn, int numExamples) {
        this.fn = fn;
        this.numExamples = numExamples;
    }
    
    @Override
    public double getValue(IntDoubleVector point, int[] batch) {
        return fn.getValue(point);
    }

    @Override
    public IntDoubleVector getGradient(IntDoubleVector point, int[] batch) {
        return fn.getGradient(point);
    }
    
    @Override
    public ValueGradient getValueGradient(IntDoubleVector point, int[] batch) {
        return fn.getValueGradient(point);
    }

    @Override
    public int getNumDimensions() {
        return fn.getNumDimensions();
    }
    
    @Override
    public int getNumExamples() {
        return numExamples;
    }

    public void setNumExamples(int numExamples) {
        this.numExamples = numExamples;
    }

}
