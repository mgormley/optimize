package edu.jhu.hlt.optimize;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.log4j.Logger;

import edu.jhu.hlt.util.math.GPRegression;
import edu.jhu.hlt.util.math.GPRegression.GPRegressor;
import edu.jhu.hlt.util.math.Kernel;
import edu.jhu.hlt.util.math.Vectors;

/**
 * Gaussian Process Global Optimization (GPGO). 
 * 
 * This implementation is based on:
 * 
 *  Bayesian Gaussian processes for sequential prediction, optimisation and quadrature.
 *  M. A. Osborne (2010).  
 *  DPhil thesis, University of Oxford. 
 *  
 * Wishlist:
 * 	- Hyperparameter estimation
 * 
 * @author noandrews
 */
public class GPGO extends    Optimizer<Function>
                  implements Maximizer<Function>, 
                             Minimizer<Function> {

	static Logger log = Logger.getLogger(GPGO.class);
	
	// Settings
	static final int order = 1; // up to what order derivatives to compute
	
	// Observations
	RealMatrix X;
	RealVector y;
	double noise;
	
	// Prior
	Kernel prior;
	RealMatrix K;
	
	// Posterior
	GPRegressor reg;
	
	// Loss function
	ExpectedMyopicLoss loss = new ExpectedMyopicLoss();
	
	public GPGO(Function f, Kernel prior) {
		super(f);
		this.prior = prior;
	}
	
	public GPGO(Function f, Kernel prior, RealMatrix X, RealVector y, double noise) {
		this(f, prior);
		this.X = X;
		this.y = y;
	}
	
	public void estimatePosterior() {
		this.reg = getPosterior();
	}
	
	public GPRegressor getPosterior() {
		return GPRegression.trainRegressor(X, y, prior, noise);
	}
	
	public GPRegressor getRegressor() {
		return reg;
	}
	
	// TODO: this should just return a Function; its mostly here for debugging
	public ExpectedMyopicLoss getExpectedLoss() {
		return loss;
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
		
		// return phi(x) = standard Gaussian pdf
	    public DerivativeStructure phi(DerivativeStructure x) {
	    	DerivativeStructure numer = x.pow(2d).negate().divide(2d).exp();
	        return numer.divide(2d * Math.PI);
	    }

	    // return phi(x, mu, signma) = Gaussian pdf with mean mu and stddev sigma
	    public DerivativeStructure phi(DerivativeStructure x, DerivativeStructure mu, DerivativeStructure sigma) {
	        return phi(x.subtract(mu).divide(sigma));
	    }
		
		// return Phi(z) = standard Gaussian cdf using a Taylor approximation
	    public DerivativeStructure Phi(DerivativeStructure z) {
	        if (z.getValue() < -8.0) return new DerivativeStructure(z.getFreeParameters(), z.getOrder(), 0d);
	        if (z.getValue() >  8.0) return new DerivativeStructure(z.getFreeParameters(), z.getOrder(), 1d);
	        DerivativeStructure sum  = new DerivativeStructure(z.getFreeParameters(), z.getOrder(), 0);
	        DerivativeStructure term = new DerivativeStructure(z.getFreeParameters(), z.getOrder(), z.getValue());
	     
	        int i=3;
	        boolean done = false;
	        while(!done) {
	        	sum = sum.add(term);
	        	term = term.multiply(z).multiply(z).divide(i);
	        	if(sum.getValue()+term.getValue() == sum.getValue()) 
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
	    
	    public DerivativeStructure predictive_mean(DerivativeStructure [] x_star) {
	    	RealVector alpha = reg.getAlpha();
	    	DerivativeStructure [] k_star = new DerivativeStructure[alpha.getDimension()];
	    	
	    	// O(n) for n function evaluations
	    	for(int i=0; i<reg.getL().getColumnDimension(); i++) {
	    		k_star[i] = prior.k( reg.getInput(i), x_star );
	    	}
	    	DerivativeStructure ret = new DerivativeStructure(x_star[0].getFreeParameters(), x_star[0].getOrder(), 0);
	    	
	    	// Compute the dot product between k_star and alpha, also in O(n)
	    	for(int i=0; i<alpha.getDimension(); i++) {
	    		ret = ret.add(k_star[i].multiply(alpha.getEntry(i)));
	    	}
	    	
	    	return ret;
	    }
	    
	    /**
	     * Solve Lx = b
	     * 
	     * @param L	Lower-triangular matrix
	     * @param x	Initially b; x on return
	     */
	    public void forward_substitute(RealMatrix L, DerivativeStructure [] x) {
	    	for(int i=0; i<x.length; i++) {
	    		for(int j=0; j<i; j++) {
	    			x[i] = x[i].subtract(x[j].multiply(L.getEntry(i, j)));
	    		}
	    		x[i] = x[i].divide(L.getEntry(i, i));
	    	}
	    }
	    
	    public DerivativeStructure predictive_var(DerivativeStructure [] x_star) {
	    	// Compute 
	    	DerivativeStructure [] k_star = new DerivativeStructure[reg.getL().getRowDimension()];
	    	log.info("col(L)="+reg.getL().getColumnDimension());
	    	log.info("row(L)="+reg.getL().getRowDimension());
	    	for(int i=0; i<reg.getL().getRowDimension(); i++) {
	    		RealVector x = reg.getInput(i);
	    		log.info("dim(x)="+x.getDimension()+" dim(x*)="+x_star.length);
	    		assert(x.getDimension()==x_star.length) : "dim(x)="+x.getDimension()+" dim(x*)="+x_star.length;
	    		k_star[i] = prior.k( x, x_star );
	    	}
	    	
	    	forward_substitute(reg.getL(), k_star);
	    	
	    	return prior.k(x_star, x_star).subtract(Vectors.dotProduct(k_star, k_star));
	    }
	    
	    /**
	     * Compute the expected loss of evaluating at x and keeping y=f(x) if it is our last function evaluation.
	     * 
	     * @param x	The input vector
	     * @param order The order of the input
	     * @return 	The expected loss (along with its first and second derivatives wrt x)
	     */
	    public DerivativeStructure computeExpectedLoss(RealVector x, int order) {
	    	
	    	// Initialize free variables
	    	DerivativeStructure [] vars = new DerivativeStructure[x.getDimension()];
	    	for(int k=0; k<vars.length; k++) {
	    		vars[k] = new DerivativeStructure(x.getDimension(), order, k, x.getEntry(k));
	    	}
	    	
	    	DerivativeStructure mean = predictive_mean(vars);
	    	DerivativeStructure var = predictive_var(vars);
	    		  
	    	// noa: these checked out, so the AD code at least matches the non-AD version
	    	log.info("AD mean (value) = " + mean.getValue());
	    	log.info("AD mean (real) = " + mean.getReal());
	    	log.info("AD var (value) = " + var.getValue());
	    	log.info("AD var (real); = " + var.getReal());
	    	
	    	log.info("reg mean = " + reg.predict(x).mean);
	    	log.info("reg var = " + reg.predict(x).var);
	    	
	    	DerivativeStructure min = new DerivativeStructure(x.getDimension(), order, minimumSoFar());
	    	log.info("min="+min.getValue());
	    	DerivativeStructure c = Phi(min, mean, var);
	    	double c2             = new NormalDistribution(mean.getValue(), var.getValue()).density(min.getValue());
	    	DerivativeStructure p = phi(min, mean, var);
	    	double p2             = new NormalDistribution(mean.getValue(), var.getValue()).cumulativeProbability(min.getValue());
	    	log.info("P1="+c.getValue()+" P2="+c2);
	    	log.info("p1="+p.getValue()+" p2="+p2);
	    	return min.add(c.multiply(mean.subtract(min))).subtract(var.multiply(p));
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
			
			RealVector x = new ArrayRealVector(point);
			DerivativeStructure res = computeExpectedLoss(x, order);
			
			return res.getValue();
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
