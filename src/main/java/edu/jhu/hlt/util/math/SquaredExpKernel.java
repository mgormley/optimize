package edu.jhu.hlt.util.math;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class SquaredExpKernel implements Kernel {

	double var;
	double len_scale;
	
	public SquaredExpKernel(double var, double len_scale) {
		this.var = var;
		this.len_scale = len_scale;
	}
	
	public SquaredExpKernel() {
		this.var = 1d;
		this.len_scale = 1d;
	}
	
	@Override
	public double k(RealVector x1, RealVector x2) {
		double res = 0d;
		for(int i=0; i<x1.getDimension(); i++) {
			res += Math.pow(x1.getEntry(i) - x2.getEntry(i),2)/len_scale*len_scale;
		}
		res *= -0.5;
		res = Math.exp(res);
		return var*var*res;
	}

	@Override
	public RealMatrix K(RealMatrix X) {
		RealMatrix K = MatrixUtils.createRealMatrix(X.getColumnDimension(), X.getColumnDimension());
		for(int i=0; i<X.getColumnDimension(); i++) {
			for(int j=0; j<X.getColumnDimension(); j++) {
				K.setEntry(i,j,k(X.getColumnVector(i), X.getColumnVector(j)));
			}
		}
		return K;
	}
	
}
