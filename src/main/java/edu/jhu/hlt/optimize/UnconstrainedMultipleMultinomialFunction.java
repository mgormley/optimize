package edu.jhu.hlt.optimize;

import edu.jhu.hlt.util.math.Multinomials;

/**
 * Wrapper of a MultipleMultinomialLogLikelihood which uses a softmax transform
 * on the log-probabilities of each Multinomial distribution to produce a new
 * function which has parameters ranging over the reals (i.e. the natural
 * parameters of the Multinomials).
 * 
 * The MultipleMultinomialLogLikelihood is defined as some function
 * f(x) s.t. its parameters x are log-probabilities: x_{ij} <= 0 and
 * \sum_{j=1}^{N_i} x_{ij} = 1 for all i.
 * 
 * This class creates a new function g(y) = f(cat(t(y))) s.t. y \in
 * \mathcal{R}^N where N = \sum_i N_i is the total number of parameters. We
 * define t(y) to return a new matrix of log probabilities obtained by passing
 * through a softmax transform: t(y)_{ij} = y_ij - \log \sum_{j=1}^{N_i}
 * \exp(y_ij). The function cat(x) simply concatentates the rows of a matrix x
 * to form a single vector.
 * 
 * @author mgormley
 */
public class UnconstrainedMultipleMultinomialFunction implements Function {

    private MultipleMultinomialLogLikelihood ll;
    // Just for caching.
    private double[] point;
    private double[][] logProbs;
    private int numDimensions;

    public UnconstrainedMultipleMultinomialFunction(MultipleMultinomialLogLikelihood ll) {
        this.ll = ll;
        this.logProbs = ll.getLogProbabilities();
        this.numDimensions = getNumEntries(logProbs);
    }

    @Override
    public void setPoint(double[] point) {
        this.point = point;
        updateLogProbsFromReals();
        ll.setLogProbabilities(logProbs);
    }

    @Override
    public double[] getPoint() {
        if (point == null) {
            updateRealsFromLogProbs();
        }
        return point;
    }

    private void updateLogProbsFromReals() {
        int idx=0;
        for (int i=0; i<logProbs.length; i++) {
            // Copy the real values into the log-probs array.
            for (int j=0; j<logProbs[i].length; j++) {
                logProbs[i][j] = point[idx];
                idx++;
            }
            Multinomials.normalizeLogProps(logProbs[i]);
        }
    }
    
    private void updateRealsFromLogProbs() {
        int idx=0;
        for (int i=0; i<logProbs.length; i++) {
            for (int j=0; j<logProbs[i].length; j++) {
                point[idx] = logProbs[i][j];
                idx++;
            }
        }
    }

    @Override
    public double getValue() {
        return ll.getLogLikelihood();
    }

    @Override
    public int getNumDimensions() {
        return numDimensions;
    }

    /** Gets the number of doubles comprising a ragged array. */
    private static int getNumEntries(double[][] raggedArray) {
        int size = 0;
        for (int i=0; i<raggedArray.length; i++) {
            size += raggedArray[i].length;
        }
        return size;
    }

    @Override
    public double getValue(double[] point) {
        // TODO remove.
        throw new RuntimeException();
    }

}
