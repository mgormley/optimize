package edu.jhu.hlt.optimize;

/**
 * An optimization technique for maximization.
 * 
 * @author mgormley
 * @author noandrews
 *
 */
public interface Maximizer<T extends Function> {

    /**
     * Maximizes a function starting from some initial point.
     * 
     * @param function The function to optimize.
     * @param point The input/output point. The initial point for maximization
     *            should be passed in. When this method returns this parameter
     *            will contain the point at which the maximizer terminated,
     *            possibly the maximum.
     * @return True if the optimizer terminated at a local or global optima. False otherwise.
     */
    // TODO: remove the point parameter.
	@Deprecated
    boolean maximize(T function, double[] point);
	boolean maximize();
}
