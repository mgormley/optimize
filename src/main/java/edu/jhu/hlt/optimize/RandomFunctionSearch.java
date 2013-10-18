package edu.jhu.hlt.optimize;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.ArrayRealVector;
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
 * Random function search: given a function f(x), generate
 * an N-length sequence f(x1), f(x2), ..., f(xN). All x_i 
 * are sampled uniformly subject to bounds constraints.
 * 
 * @author fmof
 */
public class RandomFunctionSearch {

    static Logger log = Logger.getLogger(RandomFunctionSearch.class);
	
    // Settings
    static final int order = 1; // up to what order derivatives to compute for the expected loss
	
    int numSamplesToTake = 10;
    Function function;
    Bounds bounds;

    // Observations
    RealMatrix X;
    RealVector y;
    double noise;
	
    // Magic numbers
    double min_delta = 1e-2; // don't allow observations too close to each other
    // otherwise you get singular matrices
    int budget = 100;        

    // Introspection
    long [] times;
    double [] guesses;
    double[][] points;
	
    public RandomFunctionSearch(Function f, Bounds bounds) {
	this.function = f;
	this.bounds = bounds;
    }
	
    public RandomFunctionSearch(Function f, Kernel prior, Bounds bounds, int budget) {
	this(f, bounds);
	this.numSamplesToTake = budget;
    }
	
    public RandomFunctionSearch(Function f, Kernel prior, Bounds bounds, RealMatrix X, RealVector y) {
	this(f, bounds);
	this.X = X;
	this.y = y;
    }
	
    boolean sample(){
	return sample(numSamplesToTake);
    }

    /**
     * Main method with the search loop
     * 
     * @param numTimes
     * @return
     */
    boolean sample(int numTimes){
	// Initialization
	RealVector x;	    
	// Initialize storage for introspection purposes
	times = new long[numTimes];
	guesses = new double[numTimes];
	points = new double[numTimes][];
	long startTime = System.currentTimeMillis();
	long currTime;
	for(int iter=0;iter<numTimes;iter++){
	    points[iter] = getInitialPointArray();
	    x= new ArrayRealVector(points[iter]);
	    function.setPoint(x.toArray());
	    double y = function.getValue(x.toArray());
	    //updateObservations(x, y);				
	    // Take (x,y) and add it to observations
	    currTime = System.currentTimeMillis();
	    times[iter] = currTime - startTime;
	    guesses[iter] = y;
	}		
	return true;
    }
	
    // This is needlessly inefficient: should just store a list of vectors
    private void updateObservations(RealVector x, double fx) {
	//RealMatrix new_X = new RealMatrix(X.getColumnDimension()+1, X.getRowDimension());
	System.out.println("creating...");
	RealMatrix X_new = X.createMatrix(X.getColumnDimension()+1, X.getRowDimension());
	System.out.println("done!");
	for(int i=0; i<X.getColumnDimension(); i++) {
	    X_new.setColumnVector(i, X.getColumnVector(i));
	}
	X_new.setColumnVector(X.getColumnDimension(), x);
	y.append(fx);
    }
	
    private RealVector getInitialPoint() {
	return new ArrayRealVector(getInitialPointArray());
    }
    private double[] getInitialPointArray() {
	double [] pt = new double[function.getNumDimensions()];
	// Random starting location
	for(int i=0; i<pt.length; i++) {
	    double r  = Prng.nextDouble(); //r ~ U(0,1)
	    //pt[i] = (bounds.getUpper(i)-bounds.getLower(i))*(r-1.0) + bounds.getUpper(i);
	    pt[i] = this.bounds.transformFromUnitInterval(i,r);
	}
	return pt;
    }
	
    // Introspection
    public double [] getBestGuessPerIteration() {
	return guesses;
    }
	
    public long [] getCumulativeMillisPerIteration() {
	return times;
    }

    public double[][] getPoints(){
	return points;
    }

    private double minimumSoFar() {
	double min = Double.POSITIVE_INFINITY;
	for(int i=0; i<y.getDimension(); i++) {
	    double d = y.getEntry(i);
	    if(d<min) min=d;
	}
	return min;
    }
}
