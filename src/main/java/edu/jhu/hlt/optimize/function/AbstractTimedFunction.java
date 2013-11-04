package edu.jhu.hlt.optimize.function;

/**
 * @author noandrews
 */
public abstract class AbstractTimedFunction implements TimedFunction {
	
	@Override
    public double getValue() {
        return getValue(getTime());
    }

    @Override
    public abstract int getNumDimensions();

    @Override
    public abstract void setPoint(double[] point);

    @Override
    public abstract double getValue(double t);
    
}
