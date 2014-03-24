package edu.jhu.hlt.util.stats;

import org.apache.commons.math3.linear.RealMatrix;

public interface Regressor {
	public void fit(RealMatrix X, RealMatrix y);
	public Regression predict(RealMatrix X);
}
