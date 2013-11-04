package edu.jhu.hlt.optimize.function;

import edu.jhu.prim.vector.IntDoubleVector;

/**
 * Immutable carrier class for a value and a gradient.
 * 
 * @author mgormley
 */
public class ValueGradient {

    private double value;
    private IntDoubleVector gradient;
    
    public ValueGradient(double value, IntDoubleVector gradient) {
        this.value = value;
        this.gradient = gradient;
    }

    public double getValue() {
        return value;
    }

    public IntDoubleVector getGradient() {
        return gradient;
    }    
    
}
