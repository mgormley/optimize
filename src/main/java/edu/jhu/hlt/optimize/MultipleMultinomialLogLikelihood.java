package edu.jhu.hlt.optimize;

/**
 * The MultipleMultinomialLogLikelihood is defined as some function f(x) s.t.
 * its parameters x are log-probabilities: x_{ij} <= 0 and \sum_{j=1}^{N_i}
 * x_{ij} = 1 for all i.
 * 
 * @author mgormley
 * @author fmof
 */
public interface MultipleMultinomialLogLikelihood {

    /**
     * Gets the log-probabilities.
     */
    double[][] getLogProbabilities();

    /**
     * Sets the log-probabilities..
     */
    void setLogProbabilities(double[][] logProbs);

    /**
     * Gets the log-likelihood for the current log-probabilities.
     */
    double getLogLikelihood();

}
