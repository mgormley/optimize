package edu.jhu.hlt.util.math;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public interface Kernel {

	public double k(RealVector x1, RealVector x2);
	public RealMatrix K(RealMatrix x);             // TODO: this should be something more specific than a dense matrix
	
}
