package edu.jhu.hlt.optimize;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.log4j.Logger;

import edu.jhu.hlt.util.Prng;
import edu.jhu.hlt.util.math.GPRegression;
import edu.jhu.hlt.util.math.GPRegression.GPRegressor;
import edu.jhu.hlt.util.math.GPRegression.RegressionResult;
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
	final int order = 1;            // up to what order derivatives to compute for the expected loss
	
	final boolean use_VFSA = false; // if VFSA is used to pick starting points for expected loss
									// minimization
	
	final int depth = 100;          // how many iterations of local optimization
	
	final int width_mult = 10;      // do local searches from dim(f) * width_mult start locations
	
	// Observations
	RealMatrix X;
	RealVector y;
	double noise;
	
	// Prior
	Kernel prior;
	RealMatrix K;
	
	// Posterior
	GPRegressor reg;
	
	// Loss function: minimized to select the next point to evaluate
	ExpectedMyopicLoss loss;
	
	// Magic numbers
	double min_delta = 1e-2; // don't allow observations too close to each other
	                         // otherwise you get singular matrices
	
	// How many function evaluations we are allowed
	int budget = 100;

	// Introspection
	long [] times;
	double [] guesses;
	
	/**
	   WARNING: this will not initialize X. This is because without data,
	   X will be filled in automatically. optimize accounts for this.
	   This constructor will safely initialize y though.
	 */
	// FIXME: don't take in bounds, but a BoundedFunction instead
	public GPGO(ConstrainedFunction f, Kernel prior) {
		super(f);
		this.prior = prior;
		this.loss = new ExpectedMyopicLoss(f.getNumDimensions(), f.getBounds());
		furtherInit();
	}

	/**
	   WARNING: this will not initialize X. This is because without data,
	   X will be filled in automatically. optimize accounts for this.
	   This constructor will safely initialize y though.
	 */	
	public GPGO(ConstrainedFunction f, Kernel prior, int budget) {
		this(f, prior);
		this.budget = budget;
		furtherInit();
	}
	
	public GPGO(ConstrainedFunction f, Kernel prior, RealMatrix X, RealVector y, double noise) {
		this(f, prior);
		this.X = X;
		this.y = y;
	}

	private void furtherInit(){
	    //note that we can't created X yet, because it will be filled in automatically
	    //with zeros, thus giving us off-by-one errors (it will seem as though we have
	    //one more observation than we actually do!!!
	    // if(this.X==null){
	    // 	//X is a matrix of (numDimensions X numDataPoints)

	    // 	X = MatrixUtils.createRealMatrix(this.f.getNumDimensions(),1);
	    // }
	    if(this.y==null){
		this.y=new ArrayRealVector();
	    }
	}
	
	/**
	 * Main method with the optimization loop
	 * 
	 * @param minimize
	 * @return
	 */
	boolean optimize(boolean minimize) {
		// Initialization
		RealVector x = getInitialPoint();
		double[] xarr = x.toArray();
		f.setPoint(xarr);
		double y = f.getValue(xarr);
		if(X==null){
		    X = MatrixUtils.createRealMatrix(new double[][]{xarr}).transpose();
		    this.y = this.y.append(y);
		} else {
		    updateObservations(x, y);
		}
		
		// Initialize storage for introspection purposes
		times = new long[budget];
		guesses = new double[budget];

		long startTime = System.currentTimeMillis();
		long currTime;
		
		for(int iter=0; iter<budget; iter++) {
			
			// Compute the GP posterior
			estimatePosterior();
			
			// Pick the next point to evaluate
			RealVector min = minimizeExpectedLoss();
			
			// Take (x,y) and add it to observations
			f.setPoint(min.toArray());
			y = f.getValue();
			
			currTime = System.currentTimeMillis();
			times[iter] = currTime - startTime;
			guesses[iter] = minimumSoFar();
			
			updateObservations(min, y);
		}
		
		return true;
	}
	
	public RealVector minimizeExpectedLoss() {

		double [] best_x = null;
		double best_y = Double.POSITIVE_INFINITY;

		List<RealVector> pts = getPointsToProbe();
		
		for(RealVector pt : pts) {
			ConstrainedGradientDescentWithLineSearch opt = new ConstrainedGradientDescentWithLineSearch(this.depth);
			opt.minimize(loss, pt.toArray());
			double [] x = loss.getPoint();
			double y = loss.getValue();
			if(y<best_y) {
				best_x = x;
				best_y = y;
			}
		}

		return new ArrayRealVector(best_x);
	}
	
	// This is needlessly inefficient: should just store a list of vectors
	private void updateObservations(RealVector x, double fx) {
		RealMatrix X_new = X.createMatrix(X.getRowDimension(), X.getColumnDimension()+1);
		final int numCols = X.getColumnDimension();
		for(int i=0; i<numCols; i++) {
			X_new.setColumnVector(i, X.getColumnVector(i));
		}
		X_new.setColumnVector(numCols, x);
		this.y = this.y.append(fx);
		this.X = null;
		this.X=X_new;
	}
	
	private RealVector getInitialPoint() {
		double [] pt = new double[f.getNumDimensions()];
		// Random starting location
		double effective_upper, effective_lower;
		for(int i=0; i<pt.length; i++) {
		    double r  = Prng.nextDouble(); //r ~ U(0,1)
		    pt[i] = loss.getBounds().transformFromUnitInterval(i,r);
		}
		return new ArrayRealVector(pt);
	}
	
	// Introspection
	public double [] getBestGuessPerIteration() {
		return guesses;
	}
	
	public long [] getCumulativeMillisPerIteration() {
		return times;
	}
	
	public void estimatePosterior() {
		this.reg = getPosterior();
	}
	
	public GPRegressor getPosterior() {
		assert(X.getColumnDimension() == y.getDimension());
		return GPRegression.trainRegressor(X, y, prior, noise);
	}
	
	public GPRegressor getRegressor() {
		return reg;
	}
	
	// TODO: mostly here for debugging
	public ExpectedMyopicLoss getExpectedLoss() {
		return loss;
	}

	public double minimumSoFar() {
		double min = Double.POSITIVE_INFINITY;
		for(int i=0; i<y.getDimension(); i++) {
			double d = y.getEntry(i);
			if(d<min) min=d;
		}
		return min;
	}
	
	/**
	 * @return	A list of points. A local search will be initialized from each of the points.
	 */
	public List<RealVector> getPointsToProbe() {
		List<RealVector> points = new ArrayList<RealVector>();
		
		if(use_VFSA) {
			
			// FIXME: needs testing / debugging
			VFSAOptimizer opt = new VFSAOptimizer(loss);
			opt.minimize();
			double [] pt = f.getPoint();
			points.add( new ArrayRealVector(pt) );
			
		} else {
			
			Bounds bounds = loss.getBounds();
			for(int i=0; i<this.width_mult * f.getNumDimensions(); i++) {
				double [] pt = new double[f.getNumDimensions()];
				// Random starting location
				for(int k=0; k<pt.length; k++) {
				    double r  = Prng.nextDouble(); //r ~ U(0,1)
				    pt[k] = (bounds.getUpper(k)-bounds.getLower(k))*(r-1.0) + bounds.getUpper(k);
				}
				points.add( new ArrayRealVector(pt) );
			}
			
		}
		
		return points;
	}
	
	public class ExpectedMyopicLoss implements ConstrainedDifferentiableFunction {
		
		int n;           // dimensionality
		double [] point; // storage for the current input point
		Bounds bounds;
		
		public ExpectedMyopicLoss(int n, Bounds bounds) {
			this.n = n;
			this.point = new double[n];
			this.bounds = bounds;
		}
		
		// return phi(x) = standard Gaussian pdf
	    public DerivativeStructure phi(DerivativeStructure x) {
	    	DerivativeStructure numer = x.pow(2).negate().divide(2).exp();
	        return numer.divide(Math.sqrt(2d * Math.PI));
	    }

	    // return phi(x, mu, sigma) = Gaussian pdf with mean mu and stddev sigma
	    public DerivativeStructure phi(DerivativeStructure x, DerivativeStructure mu, DerivativeStructure sigma) {
	        return phi(x.subtract(mu).divide(sigma)).divide(sigma);
	    }
		
		// return Phi(z) = standard Gaussian cdf
	    public DerivativeStructure Phi(DerivativeStructure z) {
	    	// constants
	        double a1 =  0.254829592;
	        double a2 = -0.284496736;
	        double a3 =  1.421413741;
	        double a4 = -1.453152027;
	        double a5 =  1.061405429;
	        double p  =  0.3275911;

	        // Save the sign of x
	        int sign = 1;
	        if (z.getValue() < 0)
	            sign = -1;
	        z = z.abs().divide(Math.sqrt(2.0));

	        // A&S formula 7.1.26
	        DerivativeStructure t = z.multiply(p).add(1.0).pow(-1);
	        DerivativeStructure c = t.multiply(a5).add(a4).multiply(t).add(a3).multiply(t).add(a2).multiply(t).add(a1);
	        DerivativeStructure y = z.pow(2).negate().exp().multiply(t).multiply(c).negate().add(1);

	        return y.multiply(sign).add(1).multiply(0.5);
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
	    		//log.info( "x1="+reg.getInput(i).getEntry(0) );
	    		//log.info( "x2="+x_star[0].getValue());
	    		k_star[i] = prior.k( reg.getInput(i), x_star );
	    		//log.info("[AD] k_star["+i+"]="+k_star[i].getValue());
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
	    	DerivativeStructure [] k_star = new DerivativeStructure[reg.getL().getColumnDimension()];
	    	//log.info("col(L)="+reg.getL().getColumnDimension());
	    	//log.info("row(L)="+reg.getL().getRowDimension());
	    	for(int i=0; i<reg.getL().getColumnDimension(); i++) {
	    		RealVector x = reg.getInput(i);
	    		//log.info("dim(x)="+x.getDimension()+" dim(x*)="+x_star.length);
	    		assert(x.getDimension()==x_star.length) : "dim(x)="+x.getDimension()+" dim(x*)="+x_star.length;
	    		k_star[i] = prior.k( x, x_star );
	    	}
	    	
	    	forward_substitute(reg.getL(), k_star);
	    	
	    	return prior.k(x_star, x_star).subtract(Vectors.dotProduct(k_star, k_star));
	    }
	    
	    /**
	     * Compute the expected loss of evaluating at x and keeping y=f(x) if it is our last function evaluation.
	     * 
	     * @param x		The input vector
	     * @param order Up to what order derivatives to compute
	     * @return 		The expected loss (along with its first and second derivatives wrt x)
	     */
	    public DerivativeStructure computeExpectedLoss(RealVector x, int order) {
	    	
	    	// Initialize free variables
	    	DerivativeStructure [] vars = new DerivativeStructure[x.getDimension()];
	    	for(int k=0; k<vars.length; k++) {
	    		vars[k] = new DerivativeStructure(x.getDimension(), order, k, x.getEntry(k));
	    		//log.info("[AD] x["+k+"]="+vars[k].getValue());

	    	}
	    	
	    	// Compute GP posterior mean and variance at x
	    	DerivativeStructure mean = predictive_mean(vars);
	    	DerivativeStructure var = predictive_var(vars);
	    	
	    	log.info("[AD loss] mean = " + mean.getValue());
	    	log.info("[AD loss] var = " + var.getValue());
	    		  
	    	// Get function minimum found so far
	    	DerivativeStructure min = new DerivativeStructure(x.getDimension(), order, minimumSoFar());

	    	// Compute CDF and PDF at the minimum
	    	DerivativeStructure cdf = Phi(min, mean, var);
	    	DerivativeStructure pdf = phi(min, mean, var);

	    	// Return the expected myopic loss
	    	return min.add(cdf.multiply(mean.subtract(min))).subtract(var.multiply(pdf));
	    }
	    
	    public double computeExpectedLoss(RealVector x) {
	    	
	    	for(int k=0; k<x.getDimension(); k++) {
	    		log.info("x["+k+"]="+x.getEntry(k));
	    	}
	    	
	    	// Compute GP posterior mean and variance at x
	    	RegressionResult res = reg.predict(x);
	    	double mean = res.mean;
	    	double var = res.var;
	    	
	    	log.info("[loss] mean = " + mean);
	    	log.info("[loss] var = " + var);
	    	
	    	assert(var > 0);
	    	
	    	// Get function minimum found so far
	    	double min = minimumSoFar();
	    	
	    	//log.info("min = " + min);
	    	
	    	// Compute CDF and PDF
	    	NormalDistribution N = new NormalDistribution(mean, var);
	    	double cdf = N.cumulativeProbability(min);
	    	double pdf = N.density(min);
	    	
	    	//log.info("cdf = " + cdf);
	    	//log.info("pdf = " + pdf);
	    	//log.info("mean - min = " + (mean-min));
	    	
	    	return min + (mean-min)*cdf - var*pdf;
	    }
		
		@Override
		public void getGradient(double[] gradient) {
			DerivativeStructure value = computeExpectedLoss(new ArrayRealVector(this.point), 1);
			for(int i=0; i<n; i++) {
				int [] orders = new int[n];
				orders[i] = 1;
				gradient[i] = value.getPartialDerivative(orders);
			}
		}

		@Override
		public void setPoint(double[] point) {
			for(int i=0; i<point.length; i++) {
				this.point[i] = point[i];
			}
		}

		@Override
		public double[] getPoint() {
			return point;
		}

		@Override
		public double getValue(double[] point) {
			
			RealVector x = new ArrayRealVector(point);
			DerivativeStructure res = computeExpectedLoss(x, order);
			
			return res.getValue();
		}

		@Override
		public double getValue() {
			return getValue(point);
		}

		@Override
		public int getNumDimensions() {
			return n;
		}

		@Override
		public Bounds getBounds() {
			return bounds;
		}

		@Override
		public void setBounds(Bounds b) {
			this.bounds = b;
		}

		
	}
	
	@Override
	public boolean minimize(Function function, double[] initial) {
		
		this.f = function;
		this.f.setPoint(initial);
		
		return optimize(true);
	}

	@Override
	public boolean minimize() {
		return optimize(true);
	}

	@Override
	public boolean maximize(Function function, double[] point) {
		
		this.f = function;
		this.f.setPoint(point);
		
		return optimize(false);
	}

	@Override
	public boolean maximize() {
		// TODO Auto-generated method stub
		return optimize(false);
	}
}
