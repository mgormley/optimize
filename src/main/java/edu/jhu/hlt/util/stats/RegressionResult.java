package edu.jhu.hlt.util.stats;

@Deprecated
public class RegressionResult {
	public double mean;
	public double var;
	public RegressionResult(double mean, double var) {
		this.mean = mean;
		this.var = var;
	}
}