package edu.jhu.hlt.util.stats;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import edu.jhu.hlt.optimize.function.Function;
import edu.jhu.hlt.util.Prng;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * 
 * A collection of handy stats methods.
 * 
 * @author noandrews
 * 
 */
public class Stats {
	
	public static class SampleLikelihood {
		double [] theta;
		double value;
		public SampleLikelihood(double [] theta, double value) {
			this.theta = theta;
			this.value = value;
		}
	}
	
	/**
	 * @return	Vector of iid normally distributed random variables
	 */
	public static double [] getNormalVector(int D) {
		double [] ret = new double[D];
		NormalDistribution N = new NormalDistribution(Prng.getRandom(), 0, 1, 1e-6);
		for(int i=0; i<D; i++) {
			ret[i] = N.sample();
		}
		return ret;
	}
	
	public static SampleLikelihood ellipticalSliceSampler(IntDoubleVector initial_pt, CholeskyDecomposition decomp, Function lnpdf) {
		double initial_lnpdf = lnpdf.getValue(initial_pt);
		// FIXME: what's the point of the line above?
		return ellipticalSliceSampler(initial_pt, decomp, lnpdf);
	}
	
	public static SampleLikelihood ellipticalSliceSampler(IntDoubleVector initial_pt, double initial_lnpdf, CholeskyDecomposition decomp, Function lnpdf) {	
		
		int D = initial_pt;
		RealMatrix init_val = MatrixUtils.createColumnRealMatrix(initial_pt);
		RealMatrix L = decomp.getL();
		RealMatrix r = MatrixUtils.createColumnRealMatrix(getNormalVector(D));
		RealMatrix nu = L.multiply(r);
		double hh = Math.log(Prng.nextDouble()) + initial_lnpdf;
		
		// Set up the ellipse and the slice threshold
		double phi = Prng.nextDouble()*2.0*Math.PI;
		double phi_min = phi-2.0*Math.PI;
		double phi_max = phi;
		
		// Slice sampling loop
		while(true) {
			// Compute xx for proposed angle difference and check if it's on the slice
			RealMatrix xx_prop = init_val.scalarMultiply(phi).add(nu.scalarMultiply(Math.sin(phi)));
			double cur_lnpdf = lnpdf.getValue(xx_prop.getColumn(0));
			if(cur_lnpdf > hh) {
				return new SampleLikelihood(xx_prop.getColumnVector(0).toArray(), cur_lnpdf);
			}
			// Shrink slice to rejected point
			if(phi > 0) {
				phi_max = phi;
			} else if(phi < 0) {
				phi_min = phi;
			} else {
				throw new RuntimeException("Shrunk to current position and still not acceptable");
			}
			// Propose new angle difference
			phi = Prng.nextDouble()*(phi_max - phi_min) + phi_min;
		}
	}
	
}
