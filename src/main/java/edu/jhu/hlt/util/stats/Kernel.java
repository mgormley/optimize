package edu.jhu.hlt.util.stats;

import java.util.List;

import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public interface Kernel {
	public DerivativeStructure k(DerivativeStructure [] x, DerivativeStructure [] y);
	public DerivativeStructure k(RealVector x, DerivativeStructure [] x_star);
	public double k(RealVector x1, RealVector x2);
	public RealMatrix K(RealMatrix X);            
	public List<RealMatrix> KWithPartials(RealMatrix X);
	public List<RealMatrix> getPartials(RealMatrix K);
	public void setParameters(RealVector phi);
	public RealVector getParameters();
	public int getNumParameters();
	public void grad_k(RealVector x1, RealVector x2, double [] grad);
}
