package edu.jhu.hlt.optimize.function;

/**
 * @author noandrews
 */
public abstract class AbstractDifferentiableTimedFunction extends AbstractTimedFunction implements DifferentiableTimedFunction {

	@Override
    public void getGradient(double[] gradient) {
        getGradient(getTime(), gradient);
    }

    @Override
    public abstract void getGradient(double t, double[] gradient);

    @Override
    public abstract int getNumDimensions();

    @Override
    public abstract void setPoint(double[] point);

    @Override
    public abstract double getValue(double t);
    
    @Override
    public abstract double getTime();
	
}
