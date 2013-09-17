package edu.jhu.hlt.optimize;

/**
 * An optimization technique for minimization.
 * 
 * @author mgormley
 * @author noandrews
 *
 */
public interface Minimizer<T extends Function> {

    /**
     * Minimizes a function starting from some initial point.
     * 
     * @param function The function to optimize.
     * @param point The input/output point. The initial point for minimization
     *            should be passed in. When this method returns this parameter
     *            will contain the point at which the minimizer terminated,
     *            possibly the minimum.
     * @return True if the optimizer terminated at a local or global optima. False otherwise.
     */
    // TODO: remove the point parameter.
	@Deprecated
    boolean minimize(T function, double[] initial);
    boolean minimize();
}
