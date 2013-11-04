package edu.jhu.hlt.optimize.function;


public interface Regularizer extends DifferentiableFunction {
    
    void setNumDimensions(int numParams);
    
}
