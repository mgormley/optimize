package edu.jhu.hlt.util.stats;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.log4j.Logger;

import edu.jhu.hlt.optimize.GPGO;
import edu.jhu.prim.vector.IntDoubleVector;

public class SquaredExpKernel implements Kernel {

	static Logger log = Logger.getLogger(SquaredExpKernel.class);
	
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
	public DerivativeStructure k(RealVector x, DerivativeStructure [] x_star) {
		assert(x_star.length > 0);
		assert(x.getDimension() == x_star.length);
		//log.info("len_scale = " + len_scale);
		//log.info("var = " + var);
		//log.info("[in squaredexp] x1="+x.getEntry(0));
		//log.info("[in squaredexp] x2="+x_star[0].getValue());
		DerivativeStructure res = new DerivativeStructure(x_star[0].getFreeParameters(), x_star[0].getOrder(), 0d);
		for(int i=0; i<x.getDimension(); i++) {
			res = res.add( x_star[i].subtract(x.getEntry(i)).pow(2) );
		}
		res = res.multiply(-1.0/(len_scale*len_scale));
		res = res.exp().multiply(var*var);
		//log.info("[in squaredexp] k="+res.getValue());
		return res;
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
				K.setEntry(i,j, k(X.getColumnVector(i), X.getColumnVector(j)));
			}
		}
		return K;
	}
	
	@Override
	public List<RealMatrix> KWithPartials(RealMatrix X) {
		List<RealMatrix> ret = new ArrayList<RealMatrix>();
		
		RealMatrix K = this.K(X);
		RealMatrix dvarK = K.copy();
		RealMatrix dlenK = K.copy();
		
		for(int i=0; i<K.getRowDimension(); i++) {
			for(int j=0; j<K.getColumnDimension(); j++) {
				double kij = K.getEntry(i, j);
				dvarK.setEntry(i, j, 2*(1.0/var)*kij);
				dlenK.setEntry(i, j, -Math.pow(len_scale, -3)*kij);
			}
		}
		
		ret.add(K);
		ret.add(dvarK);
		ret.add(dlenK);
		
		return ret;
	}
	

	@Override
	public List<RealMatrix> getPartials(RealMatrix K) {
		List<RealMatrix> ret = new ArrayList<RealMatrix>();
		
		RealMatrix dvarK = K.copy();
		RealMatrix dlenK = K.copy();
		
		for(int i=0; i<K.getRowDimension(); i++) {
			for(int j=0; j<K.getColumnDimension(); j++) {
				double kij = K.getEntry(i, j);
				dvarK.setEntry(i, j, 2*(1.0/var)*kij);
				dlenK.setEntry(i, j, -Math.pow(len_scale, -3)*kij);
			}
		}
		
		ret.add(dvarK);
		ret.add(dlenK);
		
		return ret;
	}
	
	@Override
	public RealVector getParameters() {
		return new ArrayRealVector(new double[] {var, len_scale});
	}
	
	@Override
	public void setParameters(RealVector phi) {
		this.var = phi.getEntry(0);
		this.len_scale = phi.getEntry(1);
	}

	@Override
	public DerivativeStructure k(DerivativeStructure[] x, DerivativeStructure[] y) {
		DerivativeStructure res = new DerivativeStructure(x[0].getFreeParameters(), x[0].getOrder(), 0d);
		for(int i=0; i<x.length; i++) {
			res = res.add( y[i].negate().add(x[i]).pow(2).divide(len_scale*len_scale) );
		}
		res = res.multiply(-0.5);
		res = res.exp();
		return res.multiply(var*var);
	}

	public void grad_k(RealVector x1, RealVector x2, double [] grad) {
		double kern = k(x1, x2);
		for(int k=0; k<x1.getDimension(); k++) {
			grad[k] = -(1.0/len_scale*len_scale)*(x2.getEntry(k)-x1.getEntry(k))*kern;
		}
	}

	@Override
	public int getNumParameters() {
		return 2;
	}

	@Override
	public void setParameters(IntDoubleVector phi) {
		this.var = phi.get(0);
		this.len_scale = phi.get(1);
	}
	
}
