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
    public double getLower(int i){
	return A[i];
    }
    public double getUpper(int i){
	return B[i];
    }
}
