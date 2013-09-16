package edu.jhu.optimize;


public interface Regularizer extends DifferentiableFunction {
    
    void setNumDimensions(int numParams);
    
}
