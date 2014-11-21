package edu.jhu.hlt.util.stats;

/**
 * Storage for regression results along with methods to evaluate.
 * 
 * @author noandrews
 */

import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Regression {
	static Logger log = LoggerFactory.getLogger(Regression.class);
	
	RealMatrix predictions;
	RealMatrix confidences;
	
	public Regression(RealMatrix predictions, RealMatrix confidences) {
		this.predictions = predictions;
		this.confidences = confidences;
	}
	
	public RealMatrix getPredictions() {
		return predictions;
	}
	
	public RealMatrix getConfidences() {
		return confidences;
	}
	
	public double meanSquaredError(RealMatrix Y) {		
		assert Y.getRowDimension() == predictions.getRowDimension();
		double total = 0;
		int n = Y.getRowDimension();
		for(int i=0; i<n; i++) {
			double y_hat = predictions.getEntry(i, 0);
			double y     = Y.getEntry(i, 0);
			total += Math.pow(y_hat-y,2);
		}
		return (1.0/n)*total;
	}
}
