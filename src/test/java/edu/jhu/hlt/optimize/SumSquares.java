package edu.jhu.hlt.optimize;

import edu.jhu.hlt.util.math.Vectors;

/**
 * The function f(x) = \sum_i (x_i + o_i)^2, where x is the current point
 * and o is a vector of offsets.
 *
 * @author mgormley
 */
public class SumSquares implements DifferentiableFunction {
    
    private int dim;
    private double[] offsets;
    private double[] point;
    
    public SumSquares(int dim) {
        this.dim = dim;
        this.offsets = new double[dim];
        this.point = new double[dim];
    }
    
    public SumSquares(double[] offsets) {
        this.dim = offsets.length;
        this.offsets = offsets;
    }
    
    @Override
    public double getValue() {
        double[] ss = new double[point.length];
        Vectors.add(ss, point);
        Vectors.add(ss, offsets);
        return Vectors.dotProduct(ss, ss);
    }

    @Override
    public void getGradient(double[] gradient) {
        for (int i=0; i<gradient.length; i++) {
            gradient[i] = 2*(point[i] + offsets[i]);
        }
    }

    @Override
    public int getNumDimensions() {
        return dim;
    }

    @Override
    public void setPoint(double[] point) {
        this.point = point;
    }
    
	@Override
	public double[] getPoint() {
		return point;
	}
    
}