package edu.jhu.hlt.optimize;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Gaussian process optimization. This implementation is based on:
 * 
 *  Bayesian Gaussian processes for sequential prediction, optimisation and quadrature.
 *  M. A. Osborne (2010).  
 *  DPhil thesis, University of Oxford. 
 * 
 * Notes:
 *  - The class is templated by a kernel (covariance) function
 * 
 * @author noandrews
 */
public class GPGO<T> extends    Optimizer<Function>
                     implements Maximizer<Function>, 
                                Minimizer<Function> {

	// Observations
	RealMatrix X;
	RealVector y;
	
	// The kernel function is used to keep K updated as new data appear
	T kernel;
	RealMatrix K;
	
	// Posterior
	RealMatrix C;
	RealVector mu;
	
	public GPGO(Function f) {
		super(f);
		// TODO Auto-generated constructor stub
	}

	private double minimumSoFar() {
		double min = Double.POSITIVE_INFINITY;
		for(int i=0; i<y.getDimension(); i++) {
			double d = y.getEntry(i);
			if(d<min) min=d;
		}
		return min;
	}
	
	public class ExpectedMyopicLoss implements TwiceDifferentiableFunction {

		GPGO<T> gp;
		
		// return phi(x) = standard Gaussian pdf
	    public DerivativeStructure phi(DerivativeStructure x) {
	    	DerivativeStructure numer = x.pow(2).negate().divide(2).exp();
	        return numer.divide(2 * Math.PI);
	    }

	    // return phi(x, mu, signma) = Gaussian pdf with mean mu and stddev sigma
	    public DerivativeStructure phi(double x, double mu, double sigma) {
	        return phi((x - mu) / sigma) / sigma;
	    }
		
		// return Phi(z) = standard Gaussian cdf using a Taylor approximation
	    public DerivativeStructure Phi(DerivativeStructure z) {
	        if (z. < -8.0) return 0.0;
	        if (z >  8.0) return 1.0;
	        double sum = 0.0, term = z;
	        for (int i = 3; sum + term != sum; i += 2) {
	            sum  = sum + term;
	            term = term * z * z / i;
	        }
	        return 0.5 + sum * phi(z);
	    }

	    // return Phi(z, mu, sigma) = Gaussian cdf with mean mu and stddev sigma
	    public DerivativeStructure Phi(double z, double mu, double sigma) {
	        return Phi((z - mu) / sigma);
	    }
		
		public ExpectedMyopicLoss(GPGO<T> gp) {
			this.gp = gp;
		}
		
		@Override
		public void getGradient(double[] gradient) {
			// TODO Auto-generated method stub
			
		}

		@Override
		public void setPoint(double[] point) {
			// TODO Auto-generated method stub
			
		}

		@Override
		public double[] getPoint() {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public double getValue(double[] point) {
			// TODO Auto-generated method stub
			return 0;
		}

		@Override
		public double getValue() {
			// TODO Auto-generated method stub
			return 0;
		}

		@Override
		public int getNumDimensions() {
			// TODO Auto-generated method stub
			return 0;
		}

		@Override
		public void getHessian(double[][] H) {
			// TODO Auto-generated method stub
			
		} 
		
	}
	
	@Override
	public boolean minimize(Function function, double[] initial) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean minimize() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean maximize(Function function, double[] point) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean maximize() {
		// TODO Auto-generated method stub
		return false;
	}
}
