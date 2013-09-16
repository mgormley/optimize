package edu.jhu.hlt.optimize;

public class Util {
	static double nanoToSec(long elapsed) {
		return (double)elapsed / 1000000000.0;
	}
	
	static double [] plusEq(double [] x, double [] y) {
		for(int k=0; k<x.length; k++) {
			x[k] += y[k];
		}
		return x;
	}
	
	/**
	 * @param c	Scale a vector by a constant 
	 * @return 	Scaled vector
	 */
	static double [] scale(double [] x, double c) {
		for(int k=0; k<x.length; k++) {
			x[k] *= c;
		}
		return x;
	}
}
