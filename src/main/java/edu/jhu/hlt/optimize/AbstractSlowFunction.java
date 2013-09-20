package edu.jhu.hlt.optimize;

import edu.jhu.hlt.util.Utilities;

public abstract class AbstractSlowFunction implements DifferentiableFunction, SlowFunction {

    protected double[] point;    
    
    public AbstractSlowFunction() {
    }

    @Override
    public void setPoint(double[] point) {
        this.point = point;
    }

    @Override
    public double getValue() {
        return this.getValue(point);
    }

    @Override
    public void getGradient(double[] gradient) {
        double[] tmp = this.getGradientAtPoint(point);
        Utilities.copy(tmp, gradient);
    }
    
    public abstract double getValue(double[] params);
    public abstract double[] getGradientAtPoint(double[] params);
    public abstract int getNumDimensions();

}
