package edu.jhu.hlt.optimize;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

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
    
    @Test
    public void testGetSetAndValue() {
        int[][] data = new int[][] {{1, 2}, {3, 4, 5}, {6, 7}};
        double[][] uniformLogProbs = new double[][] {{0.5, 0.5}, {1./3., 1./3., 1./3.}, {0.5, 0.5}};
        Vectors.log(uniformLogProbs);
        
        MockMMLL ll = new MockMMLL(data, Utilities.copyOf(uniformLogProbs));
        UnconstrainedMultipleMultinomialFunction ummf = new UnconstrainedMultipleMultinomialFunction(ll);
        
        assertEquals(7, ummf.getNumDimensions());
        assertEquals((1+2)*Math.log(0.5) + (3+4+5)*Math.log(1./3.) + (6+7)*Math.log(0.5), ummf.getValue(), 1e-13);
        ummf.setPoint(new double[]{0, 0, 0, 0, 0, 0, 0});
        JUnitUtils.assertArrayEquals(new double[]{0, 0, 0, 0, 0, 0, 0}, ummf.getPoint(), 1e-13);
        JUnitUtils.assertArrayEquals(uniformLogProbs, ll.getLogProbabilities(), 1e-13);
    }

}
