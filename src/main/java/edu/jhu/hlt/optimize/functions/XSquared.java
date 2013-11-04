package edu.jhu.hlt.optimize.functions;

import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.prim.vector.IntDoubleDenseVector;
import edu.jhu.prim.vector.IntDoubleVector;

/** 
 * The function x^2.
 * 
 * @author mgormley
 */
public class XSquared implements DifferentiableFunction {
    
	public XSquared() {  }

	@Override
	public int getNumDimensions() {
		return 1;
	}

	@Override
	public IntDoubleVector getGradient(IntDoubleVector point) {
	    return new IntDoubleDenseVector(new double[]{ 2*point.get(0) });
	}

	@Override
	public double getValue(IntDoubleVector point) {
		return point.get(0)*point.get(0);
	}

    @Override
    public ValueGradient getValueGradient(IntDoubleVector point) {
        return new ValueGradient(getValue(point), getGradient(point));
    }
	
}
