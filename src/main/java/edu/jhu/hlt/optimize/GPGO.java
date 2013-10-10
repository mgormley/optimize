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
	    public DerivativeStructure phi(DerivativeStructure x, DerivativeStructure mu, DerivativeStructure sigma) {
	        return phi(x.subtract(mu).divide(sigma));
	    }
		
		// return Phi(z) = standard Gaussian cdf using a Taylor approximation
	    public DerivativeStructure Phi(DerivativeStructure z) {
	        if (z.getReal() < -8.0) return new DerivativeStructure(z.getFreeParameters(), z.getOrder(), 0d);
	        if (z.getReal() >  8.0) return new DerivativeStructure(z.getFreeParameters(), z.getOrder(), 1d);
	        DerivativeStructure sum  = new DerivativeStructure(z.getFreeParameters(), z.getOrder(), 0);
	        DerivativeStructure term = new DerivativeStructure(z.getFreeParameters(), z.getOrder(), z.getReal());
	     
	        int i=3;
	        boolean done = false;
	        while(!done) {
	        	sum.add(term);
	        	term.multiply(z).multiply(z).divide(i);
	        	if(sum.getReal()+term.getReal() != sum.getReal()) 
	        		done = true;
	        }
	        return phi(z).multiply(sum).add(0.5);
	        
	        //for (int i = 3; sum + term != sum; i += 2) {
	        //    sum  = sum + term;
	        //    term = term * z * z / i;
	        //}
	        //return 0.5 + sum * phi(z);
	    }

	    // return Phi(z, mu, sigma) = Gaussian cdf with mean mu and stddev sigma
	    public DerivativeStructure Phi(DerivativeStructure z, DerivativeStructure mu, DerivativeStructure sigma) {
	        return Phi(z.subtract(mu).divide(sigma));
	    }
	    
	    /**
	     * Compute the expected loss of evaluating at x and keeping f(x) if it is our last function evaluation.
	     * 
	     * @param x	The input vector
	     * @return 	The expected loss (along with its first and second derivatives wrt x)
	     */
	    public DerivativeStructure computeExpectedLoss(RealVector x) {
	    	// TODO
	    	return null;
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
