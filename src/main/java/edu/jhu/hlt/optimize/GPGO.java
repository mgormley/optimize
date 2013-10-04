package edu.jhu.hlt.optimize;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Gaussian process optimization. This implementation is based on:
 * 
 *  M. A. Osborne (2010). Bayesian Gaussian processes for sequential prediction, optimisation and quadrature, 
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
