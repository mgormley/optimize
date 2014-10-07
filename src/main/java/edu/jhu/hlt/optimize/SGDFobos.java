package edu.jhu.hlt.optimize;

import org.apache.log4j.Logger;

import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.prim.list.DoubleArrayList;
import edu.jhu.prim.util.Lambda.FnIntDoubleToVoid;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * Stochastic gradient descent with forward-backward splitting (Duchi & Singer, 2009). The
 * regularizers are built into this optimizer. The key advantage of this approach is that we can
 * employ lazy updates based on the regularizers gradient, so that the gradient updates remain sparse.
 * 
 * In the case of L1 regularization, this is also called truncated gradient (Langford et al., 2008).
 * 
 * @author mgormley
 */
public class SGDFobos extends SGD implements Optimizer<DifferentiableBatchFunction> {

    /** Options for this optimizer. */
    public static class SGDFobosPrm extends SGDPrm {
        private static final long serialVersionUID = 1L;
        /** The weight on the l1 regularizer. */
        public double l1Lambda = 0.0;
        /** The weight on the l2^2 regularizer. */
        public double l2Lambda = 0.0;
    }

    private static final Logger log = Logger.getLogger(SGDFobos.class);

    private final SGDFobosPrm prm;
    // Whether to use l1 or l2^2 regularization.
    private final boolean l1reg;
    // The iteration of the last step taken for each model parameter.
    private double[] iterOfLastStep;
    // Cumulative learning rates.
    private DoubleArrayList accumLr;
    
    /**
     * Constructs an SGD optimizer.
     */
    public SGDFobos(SGDFobosPrm prm) {
        super(prm);
        if (prm.l1Lambda != 0 && prm.l2Lambda != 0) {
            throw new IllegalArgumentException("Only one of L1 or L2 regularization may be used");
        }
        if (!prm.sched.isSameForAllParameters()) {
            throw new IllegalArgumentException("Fobos requires that the gain schedule is the same for all parameters");
        }
        this.l1reg = (prm.l1Lambda != 0);
        this.prm = prm;
    }
    
    /**
     * Initializes all the parameters for optimization.
     */
    protected void init(DifferentiableBatchFunction function) {
        super.init(function);
        this.iterOfLastStep = new double[function.getNumDimensions()];
        while (accumLr.size() < iterCount) {
            accumLr.add(prm.sched.getLearningRate(iterCount, 0));
        }
    }

    protected void takeGradientStep(final IntDoubleVector point, final IntDoubleVector gradient, final boolean maximize) {
        // We always assume that the schedule is the same for all parameters.
        double lr_t = prm.sched.getLearningRate(iterCount, 0);
        gradient.iterate(new FnIntDoubleToVoid() {
            @Override
            public void call(int index, double gr) {
                double lr_0 = prm.sched.getLearningRate(iterOfLastStep[index], 0);
                double w_0 = point.get(index);
                // Step 1. Eq (2) from Duchi & Singer (2009)
                double w_1 = maximize ?  w_0 + lr * gr : w_0 - lr * gr;
                // Step 2. Eq (3) from Duchi & Singer (2009)
                double w_2;
                if (l1reg) {
                    // l1 regularization. 
                    // Eq (19) from Duchi & Singer (2009)
                    w_2 = (w_1 < 0 ? -1 : 1) * Math.max(0, Math.abs(w_1) - lr*prm.l1Lambda);
                } else {
                    // l2^2 regularization.
                    // Eq. (20) from Duchi & Singer (2009).
                    w_2 = w_1 / (1 + lr*prm.l2Lambda);
                }
                assert !Double.isNaN(w_2);
                assert !Double.isInfinite(w_2);
                point.set(index, w_2);
                iterOfLastStep[index] = iterCount;
            }
        });
    }

}
