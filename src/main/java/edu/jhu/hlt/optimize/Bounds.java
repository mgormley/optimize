package edu.jhu.hlt.optimize;

/**
 * This should probably be part of a specialized Function class.
 * 
 * @author noandrews
 */
public class Bounds {
	public double [] A; // lower bounds
	public double [] B; // upper bounds
	
	public Bounds(double [] A, double [] B) {
		this.A = A;
		this.B = B;
	}

    public Bounds(int dim){
    	this.A=new double[dim];
		this.B = new double[dim];
    }
    
    public double getLower(int i) {
    	return A[i];
    }
    
    public double getUpper(int i) {
    	return B[i];
    }

    public boolean inBounds(double [] pt) {
    	for(int i=0; i<pt.length; i++) {
    		if (pt[i] > B[i]) return false;
    		if (pt[i] < A[i]) return false;
    	}
    	return true;
    }
    
    public static Bounds getUnitBounds(int dim) {
    	double [] L = new double[dim];
    	double [] U = new double[dim];
    	for(int i=0; i<dim; i++) {
    		L[i] = 0;
    		U[i] = 1;
    	}
    	return new Bounds(L, U);
    }
    
    /**
       Maps d \in [0,1] to transformed(d), such that
       transformed(d) \in [A[i], B[i]]. The transform is 
       just a linear one. It does *NOT* check for +/- infinity.
     */
    public double transformFromUnitInterval(int i, double d){
    	return (B[i]-A[i])*(d-1.0)+B[i];
    }

    /**
       Maps d \in [l,u] to transformed(d), such that
       transformed(d) \in [A[i], B[i]]. The transform is 
       just a linear one. It does *NOT* check for +/- infinity.
     */
    public double transformRangeLinearly(double l, double u, int i, double d){
    	return (B[i]-A[i])/(u-l)*(d-u)+B[i];
    }
}
