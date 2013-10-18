package edu.jhu.hlt.optimize;

/**
 * A function f defined on a vector x of m multinomials,
 * which can be partitioned into m contiguous blocks.
 *
 * @author fmof
 */
public interface MultipleMultinomialLogLikelihood{
    
    /**
       Returns an array of length m denoting the (zero-indexed)
       starting positions of each m multinomial. If the underlying 
       point x=(0.9, 0.1, 0.2, 0.2, 0.6) represents two multinomials 
       m1=(0.9,0.1) and m2=(0.2, 0.2, 0.6), then getBoundaries() returns
       {0, 2}. Note that in general the resulting multinomials may need
       to be transformed to respect boundary and sum-to-one constraints.
     */
    int[] getBoundaries();

    /**
       Returns the log-probability representation of the m multinomials.
     */
    double[][] getLogProbabilities();
    
    /**
       Set the log-probabilities. Must be linearizable according to 
       getBoundaries()
    */
    void setLogProbabilities(double[][] logProbs);
}
