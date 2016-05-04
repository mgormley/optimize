package edu.jhu.hlt.optimize.functions;

import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.prim.vector.IntDoubleDenseVector;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * The function f(x) = \sum_{i=1}^n i * (x_i)^2, where x is the current point.
 * 
 * Remarks from http://www.robertmarks.org/Classes/ENGR5358/Papers/functions.pdf:
 * 
 * Test area is usually restricted to hyphercube −5.12 ≤ xi ≤ 5.12, i = 1, . . . , n. Its global
 * minimum equal f(x) = 0 is obtainable for xi = 0, i = 1, . . . , n.
 *
 * @author mgormley
 */
public class WeightedSphereModel implements DifferentiableFunction {
    
    private int dim;
    
    public WeightedSphereModel(int dim) {
        this.dim = dim;
    }
    
    public WeightedSphereModel(double[] offsets) {
        this.dim = offsets.length;
    }
    
    @Override
    public double getValue(IntDoubleVector point) {
        double val = 0;
        for (int i=0; i<dim; i++) {
            val += (i+1) * point.get(i) * point.get(i);
        }
        return val;
    }

    @Override
    public IntDoubleVector getGradient(IntDoubleVector point) {
        IntDoubleDenseVector gradient = new IntDoubleDenseVector(dim);
        for (int i=0; i<dim; i++) {
            gradient.set(i, (i+1) * 2 * point.get(i));
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