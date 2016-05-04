package edu.jhu.hlt.optimize.functions;

import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.prim.vector.IntDoubleDenseVector;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * The function f(x) = \sum_i (x_i + o_i)^2, where x is the current point
 * and o is a vector of offsets.
 *
 * @author mgormley
 */
public class SumSquares implements DifferentiableFunction {
    
    private int dim;
    private double[] offsets;
    
    public SumSquares(int dim) {
        this.dim = dim;
        this.offsets = new double[dim];
    }
    
    public SumSquares(double[] offsets) {
        this.dim = offsets.length;
        this.offsets = offsets;
    }
    
    @Override
    public double getValue(IntDoubleVector point) {
        IntDoubleDenseVector ss = new IntDoubleDenseVector(dim);
        ss.add(point);
        ss.add(new IntDoubleDenseVector(offsets));
        return ss.dot(ss);
    }

    @Override
    public IntDoubleVector getGradient(IntDoubleVector point) {
        IntDoubleDenseVector gradient = new IntDoubleDenseVector(dim);
        for (int i=0; i<dim; i++) {
            gradient.set(i, 2*(point.get(i) + offsets[i]));
        }
        return gradient;
    }

    @Override
    public ValueGradient getValueGradient(IntDoubleVector point) {
        return new ValueGradient(getValue(point), getGradient(point));
    }

    @Override
    public int getNumDimensions() {
        return dim;
    }
    
}