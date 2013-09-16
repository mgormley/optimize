package edu.jhu.hlt.optimize;


public interface Regularizer extends DifferentiableFunction {
    
    void setNumDimensions(int numParams);
    
}
