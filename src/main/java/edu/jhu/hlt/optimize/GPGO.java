package edu.jhu.hlt.optimize;

import java.awt.Color;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.xeiam.xchart.Chart;
import com.xeiam.xchart.ChartBuilder;
import com.xeiam.xchart.Series;
import com.xeiam.xchart.SeriesMarker;
import com.xeiam.xchart.SwingWrapper;
import com.xeiam.xchart.StyleManager.ChartType;

import edu.jhu.hlt.optimize.function.Bounds;
import edu.jhu.hlt.optimize.function.ConstrainedDifferentiableFunction;
import edu.jhu.hlt.optimize.function.ConstrainedFunction;
import edu.jhu.hlt.optimize.function.Function;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.hlt.optimize.propose.Proposable;
import edu.jhu.util.Prng;
import edu.jhu.hlt.util.math.Vectors;
import edu.jhu.hlt.util.stats.GPRegression;
import edu.jhu.hlt.util.stats.Kernel;
import edu.jhu.hlt.util.stats.GPRegression.GPRegressor;
import edu.jhu.hlt.util.stats.RegressionResult;
import edu.jhu.prim.vector.IntDoubleDenseVector;
import edu.jhu.prim.vector.IntDoubleVector;

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
 *  - Implement the Maximizer interface; for now, pass in a negated function
 *    (e.g., by using FunctionOpts.NegateFunction).
 * 	- Hyperparameter estimation
 * 
 * @author noandrews
 */
public class GPGO implements Optimizer<Function> {

	static Logger log = LoggerFactory.getLogger(GPGO.class);
	
	// Function to be optimized
	Function f;
	
	// Observations
	RealMatrix X;
	RealVector y;
	
	// Prior
	double noise = 0.01;
	Kernel prior;
	RealMatrix K;
	
	// Posterior
	GPRegressor reg;
	
	// Loss function: minimized to select the next point to evaluate
	ExpectedMyopicLoss loss;
	
	// Magic numbers
	double min_delta = 1e-2; // don't allow observations too close to each other
	double min_improve = 1e-3; // if the improvement is less than this, pick a random
	                           // point rather than that returned by the optimization
	
	int budget = 100; 	            // # of GPGO iterations

	// These next two parameters control the time-accuracy tradeoff in picking the
	// next point to evaluate.
	boolean use_SA = true;
	int width = 10000;        // # of expected loss evaluations used to find starting point for
									// local optimization
	int depth = 5;            // # of expected loss + gradient evalutaions to optimize expected loss
	
	// Introspection
	long [] times;
	double [] guesses;
	
	/**
	   WARNING: this will not initialize X. This is because without data,
	   X will be filled in automatically. optimize accounts for this.
	   This constructor will safely initialize y though.
	 */
	// FIXME: don't take in bounds, but a BoundedFunction instead
	public GPGO(ConstrainedFunction f, Kernel prior, double noise) {
		this.f = f;
		this.prior = prior;
		this.noise = noise;
		this.loss = new ExpectedMyopicLoss(f.getNumDimensions(), f.getBounds());
		furtherInit();
	}

	/**
	   WARNING: this will not initialize X. This is because without data,
	   X will be filled in automatically. optimize accounts for this.
	   This constructor will safely initialize y though.
	 */	
	public GPGO(ConstrainedFunction f, Kernel prior, double noise, int budget) {
		this(f, prior, noise);
		this.budget = budget;
		furtherInit();
	}
	
	public GPGO(ConstrainedFunction f, Kernel prior, RealMatrix X, RealVector y, double noise) {
		this(f, prior, noise);
		this.X = X;
		this.y = y;
	}

	public void setSearchParam(int width, int depth) {
		this.width = width;
		this.depth = depth;
	}
	
	private void furtherInit() {
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
	
	public void setInitialPoint() {
		// Initialization
		RealVector x = getInitialPoint();
		double[] xarr = x.toArray();
		double y = f.getValue(new IntDoubleDenseVector(xarr));
		if(X==null){
			X = MatrixUtils.createRealMatrix(new double[][]{xarr}).transpose();
			this.y = this.y.append(y);
		} else {
			updateObservations(x, y);
		}
	}
	
	public void doIter(int iter, boolean minimize) {
		int best_index = currentBestIndex(minimize);
		log.info("[GPGO] best index = " + best_index);
		double optimum_so_far = this.y.getEntry(best_index);
		log.info("[GPGO] iter = " + iter + ": " + "optimum = " + optimum_so_far);
		
		// Compute the GP posterior
		estimatePosterior();
		
		// Pick the next point to evaluate
		RealVector min = minimizeExpectedLoss();
		
		// Compute f(min) = y
		double [] min_arr = min.toArray();
		double y = f.getValue(new IntDoubleDenseVector(min_arr));
		
		log.info("l(min) = " + y);
		
		// Add (min, y) to the observations
		updateObservations(min, y);
	}
	
	/***
	 * DEBUG
	 */
	public RealVector doIterNoUpdate(int iter, boolean minimize) {
		int best_index = currentBestIndex(minimize);
		log.info("[GPGO] best index = " + best_index);
		double optimum_so_far = this.y.getEntry(best_index);
		log.info("[GPGO] iter = " + iter + ": " + "optimum = " + optimum_so_far);
		
		// Compute the GP posterior
		estimatePosterior();
		
		// Pick the next point to evaluate
		RealVector min = minimizeExpectedLoss();
		
		// Compute f(min) = y
		double [] min_arr = min.toArray();
		double y = f.getValue(new IntDoubleDenseVector(min_arr));
		
		log.info("l(min) = " + y);
		
		return min;
	}
	
	/**
	 * Main method with the optimization loop
	 * 
	 * @param minimize
	 * @return
	 */
	boolean optimize(boolean minimize, IntDoubleVector initial) {
		
		// FIXME: this ignores the given point
		
		// Set some random initial points
		setInitialPoint();
		setInitialPoint();
		setInitialPoint();
		
		// Initialize storage for introspection purposes
		times = new long[budget];
		guesses = new double[budget];

		long startTime = System.currentTimeMillis();
		long currTime;
		
		for(int iter=0; iter<budget; iter++) {			
			doIter(iter, minimize);
			
			currTime = System.currentTimeMillis();
			times[iter] = currTime - startTime;
			guesses[iter] = minimumSoFar();
		}
		
		return true;
	}
	
	// Minimize the expected loss and return the point at its estimated minimum
	public RealVector minimizeExpectedLoss() {

		double [] best_x = new double[f.getNumDimensions()];
		double best_y = Double.POSITIVE_INFINITY;

		List<RealVector> pts = getPointsToProbe();
		
		// Get the initial minimum
		for(RealVector pt : pts) {
			double l = loss.computeExpectedLoss(pt);
			if(l<best_y) {
				best_x = pt.toArray();
				best_y = l;
			}
		}
		
		double temp = best_y;
		log.info("[GPGO] Initial EL minimum prior to local search = " + best_y);
		
		// Run a local search starting from each of the returned points
		for(RealVector pt : pts) {
			GradientDescentWithLineSearch opt = new GradientDescentWithLineSearch(this.depth);
			IntDoubleDenseVector ret = new IntDoubleDenseVector(pt.toArray());
			opt.minimize(loss, ret);
			double y = loss.getValue(ret);
			log.info("[GPGO] Improvement from local search = " + (temp-y));
			if(y<best_y) {
				log.info("[GPGO] Keeping the point from local search");
				for(int i=0; i<best_x.length; i++) {
					best_x[i] = ret.get(i);
				}
				best_y = y;
			} else {
				log.info("[GPGO] Discarding the point from local search");
			}
		}
		
		return new ArrayRealVector(best_x);
	}
	
	// This is needlessly inefficient: should just store a list of vectors
	public void updateObservations(RealVector x, double fx) {
		RealMatrix X_new = X.createMatrix(X.getRowDimension(), X.getColumnDimension()+1);
		final int numCols = X.getColumnDimension();
		for(int i=0; i<numCols; i++) {
			X_new.setColumnVector(i, X.getColumnVector(i));
		}
		X_new.setColumnVector(numCols, x);
		this.y = this.y.append(fx);
		this.X = null;
		this.X = X_new;
	}
	
	private RealVector getInitialPoint() {
		
		if(f instanceof Proposable) {
			Proposable p = (Proposable)f;
			double [] pt = p.samplePoint();
			return new ArrayRealVector(pt);
		}
		
		double [] pt = new double[f.getNumDimensions()];
		// Random starting location
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
		return GPRegression.computePosterior(X, y, prior, noise);
	}
	
	public GPRegressor getRegressor() {
		return reg;
	}
	
	// TODO: mostly here for debugging
	public ExpectedMyopicLoss getExpectedLoss() {
		return loss;
	}

	public int currentBestIndex(boolean minimize) {
		double opt;
		int opt_i = -1;
		if(minimize) {
			opt = Double.POSITIVE_INFINITY;
		}
		else {
			opt = Double.NEGATIVE_INFINITY;
		}
		for(int i=0; i<y.getDimension(); i++) {
			double d = y.getEntry(i);
			log.info("y("+i+")="+d);
			if(minimize) {
				if(d<opt) {
					opt = d;
					opt_i = i;
				}
			} else {
				if(d>opt) {
					opt = d;
					opt_i = i;
				}
			}
		}
		return opt_i;
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
		
//		if(use_SA) {
//			double initial_loss = loss.getValue();
//			ParameterFreeSAOptimizer opt = new ParameterFreeSAOptimizer(loss, width);
//			opt.minimize();
//			double [] pt = loss.getPoint();
//			RealVector sa_min = new ArrayRealVector(pt);
//			
//			// Check to see if this resulted in significant improvement
//			double delta = Math.abs(initial_loss-loss.getValue());
//			if(delta < this.min_improve) {
//				// Pick a random point instead
//				log.info("SA failed to improve (delta="+delta+"), picking random point to eval");
//				RealVector x = getInitialPoint();
//				//log.info("random x="+x.getEntry(0));
//				points.add( x );
//			} else {
//				log.info("SA found minimum = " + loss.computeExpectedLoss(sa_min));
//				points.add( sa_min );
//			}	
//			
//		} else {
			
			Bounds bounds = loss.getBounds();
			for(int i=0; i<this.width; i++) {
				double [] pt = new double[f.getNumDimensions()];
				for(int k=0; k<pt.length; k++) {
				    double r  = Prng.nextDouble(); //r ~ U(0,1)
				    pt[k] = (bounds.getUpper(k)-bounds.getLower(k))*(r-1.0) + bounds.getUpper(k);
				}
				
				points.add( new ArrayRealVector(pt) );
			}
			
		//}
		
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
//	    public DerivativeStructure computeExpectedLoss(RealVector x, int order) {
//	    	
//	    	// Initialize free variables
//	    	DerivativeStructure [] vars = new DerivativeStructure[x.getDimension()];
//	    	for(int k=0; k<vars.length; k++) {
//	    		vars[k] = new DerivativeStructure(x.getDimension(), order, k, x.getEntry(k));
//	    		//log.info("[AD] x["+k+"]="+vars[k].getValue());
//
//	    	}
//	    	
//	    	// Compute GP posterior mean and variance at x
//	    	DerivativeStructure mean = predictive_mean(vars);
//	    	DerivativeStructure var = predictive_var(vars);
//	    	
//	    	log.info("[AD loss] mean = " + mean.getValue());
//	    	log.info("[AD loss] var = " + var.getValue());
//	    		  
//	    	// Get function minimum found so far
//	    	DerivativeStructure min = new DerivativeStructure(x.getDimension(), order, minimumSoFar());
//
//	    	// Compute CDF and PDF at the minimum
//	    	DerivativeStructure cdf = Phi(min, mean, var);
//	    	DerivativeStructure pdf = phi(min, mean, var);
//
//	    	// Return the expected myopic loss
//	    	return min.add(cdf.multiply(mean.subtract(min))).subtract(var.multiply(pdf));
//	    }
	    
	    public double computeExpectedLoss(RealVector x) {
	    	
	    	//log.info("computing expected loss:");
	    	//for(int k=0; k<x.getDimension(); k++) {
	    	//	log.info("x["+k+"]="+x.getEntry(k));
	    	//}
	    	
	    	//log.info("loss based on "+y.getDimension()+" observations");
	    	
	    	// Compute GP posterior mean and variance at x
	    	RegressionResult res = reg.predict(x);
	    	double mean = res.mean;
	    	double var = res.var;
	    	
	    	//log.info("[loss] mean = " + mean);
	    	//log.info("[loss] var = " + var);
	    
	    	//assert(var > 0);
	    	
	    	// Get function minimum found so far
	    	double min = minimumSoFar();
	    	
	    	//log.info("[loss] min = " + min);
	    	
	    	// Compute CDF and PDF
	    	NormalDistribution N = new NormalDistribution(mean, var);
	    	double cdf = N.cumulativeProbability(min);
	    	double pdf = N.density(min);
	    	
	    	//log.info("cdf = " + cdf);
	    	//log.info("pdf = " + pdf);
	    	//log.info("mean - min = " + (mean-min));
	    	//log.info("(mean-min)*cdf = " + ((mean-min)*cdf));
	    	//log.info("var*pdf = " + (var*pdf));
	    	
	    	return min + (mean-min)*cdf - var*pdf;
	    }
	    
	    // Should probably compute both l(x) and dx l(x) in the same
	    // method to avoid duplicate work
	    public double [] computeExpectedLossGradient(RealVector x) {
	    	double [] g = new double[x.getDimension()];
	    	RegressionResult res = reg.predict(x);
	    	double m = res.mean;
	    	double c = res.var;
	    	double stddev = Math.sqrt(c);
	    	double [] dm = new double[x.getDimension()];
	    	double [] dc = new double[x.getDimension()];
	    	reg.computeMeanGradient(x, dm);
	    	reg.computeCovarGradient(x, dc);
	    	NormalDistribution N = new NormalDistribution();
	    	double min = minimumSoFar();
	    	double std_min = (min-m)/stddev;
	    	double [] dmin = new double[x.getDimension()];
	    	for(int i=0; i<x.getDimension(); i++) {
	    		dmin[i] = (-dm[i]*stddev-(min-m)*(1.0/2*stddev)*dc[i])/c;
	    	}
	    	
	    	// XXXX Debug: Check the dmin gradient
//	    	double eps = 1e-6;
//	    	// Perturb x
//	    	RealVector x_plus_eps = x.copy();
//	    	x_plus_eps.addToEntry(0, eps); 
//	    	RegressionResult eps_res = reg.predict(x_plus_eps);
//	    	double eps_m = eps_res.mean;
//	    	double eps_c = eps_res.var;
//	    	double eps_std_min = (min-eps_m)/Math.sqrt(eps_c);
//	    	double approx_dmin = (eps_std_min - std_min) / eps;
//	    	log.info("dmin = " + dmin[0]);
//	    	log.info("approx dmin = " + approx_dmin);
//	    	// end debug
//	    			
//	    	
//	    	double eps_rhs = eps_c*N.density(eps_std_min)/Math.sqrt(eps_c);
//	    	double rhs = c*N.density(std_min)/Math.sqrt(c);
//	    	double approx_d_rhs = (eps_rhs-rhs)/eps;
//	    	double d_rhs = 0.5*(1.0/stddev)*dc[0]*N.density(std_min) - stddev*std_min*N.density(std_min)*dmin[0];
//	    	log.info("approx drhs = " + approx_d_rhs);
//	    	log.info("drhs = " + d_rhs);
//	    	
////	    	log.info("min = " + min);
////	    	log.info("m="+m);
////	    	log.info("c="+c);
	    	double pdf = N.density(std_min)/stddev;
	    	double cdf = N.cumulativeProbability(std_min);
//	    	
////	    	NormalDistribution N2 = new NormalDistribution(m, stddev);
////	    	double pdf2 = N2.density(min);
////	    	double cdf2 = N2.cumulativeProbability(min);
////	    	
////	    	log.info("normal transform:");
////	    	log.info(pdf);
////	    	log.info(pdf2);
////	    	log.info(cdf);
////	    	log.info(cdf2);
////	    	
////	    	System.exit(1);
	    	
	    	for(int i=0; i<x.getDimension(); i++) {
		    	double d2 = 0.5*(1.0/stddev)*dc[i]*N.density(std_min) - stddev*std_min*N.density(std_min)*dmin[i];
	    		g[i] = dm[i]*cdf + m*pdf*dmin[i] - min*pdf*dmin[i] - d2;
	    	}
	    	
	    	return g;
	    }
		
		@Override
		public IntDoubleVector getGradient(IntDoubleVector pt) {
			RealVector x = new ArrayRealVector(point);
			double [] g = computeExpectedLossGradient(x);
			return new IntDoubleDenseVector(g);
		}

		@Override
		public double getValue(IntDoubleVector pt) {
			double [] a = new double[f.getNumDimensions()];
			for(int i=0; i<f.getNumDimensions(); i++) {
				a[i] = pt.get(i);
			}
			return computeExpectedLoss(new ArrayRealVector(a));
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

		@Override
		public ValueGradient getValueGradient(IntDoubleVector point) {
			double val = getValue(point);
			IntDoubleVector grad = getGradient(point);
			return new ValueGradient(val, grad);
		}

		
	}
	
	@Override
	public boolean minimize(Function function, IntDoubleVector initial) {	
		this.f = function;
		return optimize(true, initial);
	}

	@Override
	public boolean maximize(Function function, IntDoubleVector initial) {
		this.f = function;
		return optimize(false, initial);
	}
}
