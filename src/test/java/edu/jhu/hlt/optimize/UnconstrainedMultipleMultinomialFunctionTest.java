package edu.jhu.hlt.optimize;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import edu.jhu.hlt.optimize.function.MultipleMultinomialLogLikelihood;
import edu.jhu.hlt.optimize.function.UnconstrainedMultipleMultinomialFunction;
import edu.jhu.hlt.util.JUnitUtils;
import edu.jhu.hlt.util.Utilities;
import edu.jhu.hlt.util.math.Vectors;

/**
 * Tests of UnconstrainedMultipleMultinomialFunction using a simple mock log-likelihood function.
 * @author mgormley
 */
public class UnconstrainedMultipleMultinomialFunctionTest {

    /**
     * A simple log-likelihood function which takes counts of each of the
     * multinomial observations as input data.
     * 
     * @author mgormley
     */
    public static class MockMMLL implements MultipleMultinomialLogLikelihood {

        int[][] data;
        double[][] logProbs;
        
        public MockMMLL(int[][] data, double[][] logProbs) {
            this.data = data;
            this.logProbs = logProbs;
        }
        
        @Override
        public double[][] getLogProbabilities() {
            return logProbs;
        }

        @Override
        public void setLogProbabilities(double[][] logProbs) {
            this.logProbs = logProbs;
        }

        @Override
        public double getLogLikelihood() {
            double ll = 0;
            for (int i=0; i<logProbs.length; i++) {
                for (int j=0; j<logProbs[i].length; j++) {
                    ll += data[i][j] * logProbs[i][j]; 
                }
            }
            return ll;
        }
        
    }

}
