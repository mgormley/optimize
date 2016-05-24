package edu.jhu.hlt.optimize;

import java.util.Arrays;

import org.apache.commons.math3.util.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.util.Prm;
import edu.jhu.prim.arrays.DoubleArrays;
import edu.jhu.prim.arrays.IntArrays;
import edu.jhu.prim.util.Lambda.FnIntDoubleToVoid;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * AdaGrad with a Composite Objective Mirror Descent update and L2 regularizer with lazy updates (Duchi et al., 2011).
 * 
 * @author mgormley
 */
public class AdaGradComidL2 extends SGD implements Optimizer<DifferentiableBatchFunction> {

    /** Options for this optimizer. */
    public static class AdaGradComidL2Prm extends SGDPrm {
        private static final long serialVersionUID = 1L;
        /** The scaling parameter for the learning rate. */
        public double eta = 0.1;
        /**
         * The amount added (delta) to the sum of squares outside the square
         * root. This is to combat the issue of tiny gradients throwing the hole
         * optimization off early on.
         */
        public double constantAddend = 1e-9;
        /** The weight on the l2 regularizer. */
        public double l2Lambda = 0.0;
        /** Initial sum of squares value. */
        public double initialSumSquares = 0;
        public AdaGradComidL2Prm() {
            // Schedule must be null.
            this.sched = null;
        }
    }

    private static final long serialVersionUID = 1L;
    private static final Logger log = LoggerFactory.getLogger(AdaGradComidL2.class);

    private final AdaGradComidL2Prm prm;
    // The iteration of the last step taken for each model parameter.
    private int[] iterOfLastStep;
    // Sum of squares of gradients up to current time for each parameter.
    private double[] gradSumSquares;
    
    /**
     * Constructs an SGD optimizer.
     */
    public AdaGradComidL2(AdaGradComidL2Prm prm) {
        super(prm);
        if (!(prm.sched == null || prm.sched instanceof EmptyGainSchedule)) {
            log.warn("Schedule for AdaGrad should be set to null. Ignoring it.");
        }
        this.prm = prm;
        this.prm.sched = new EmptyGainSchedule();
    }
    
    @Override
    public AdaGradComidL2 copy() {
        AdaGradComidL2 sgd = new AdaGradComidL2(Prm.clonePrm(this.prm));
        sgd.iterOfLastStep = IntArrays.copyOf(this.iterOfLastStep);
        sgd.gradSumSquares = DoubleArrays.copyOf(this.gradSumSquares);
        return sgd;
    }
    
    /**
     * Initializes all the parameters for optimization.
     */
    @Override
    protected void init(DifferentiableBatchFunction function, IntDoubleVector point) {
        super.init(function, point);
        this.iterOfLastStep = new int[function.getNumDimensions()];
        Arrays.fill(iterOfLastStep, -1);        
        this.gradSumSquares = new double[function.getNumDimensions()];
        Arrays.fill(gradSumSquares, prm.initialSumSquares);
    }

    @Override
    protected void takeGradientStep(final IntDoubleVector point, final IntDoubleVector gradient, 
            final boolean maximize, final int iterCount) {
        gradient.iterate(new FnIntDoubleToVoid() {
            @Override
            public void call(int i, double g_ti) {
                g_ti = maximize ? -g_ti : g_ti;
                assert !Double.isNaN(g_ti);
                // Get the old learning rate.
                double h_t0ii = prm.constantAddend + Math.sqrt(gradSumSquares[i]);
                // Update the sum of squares.
                gradSumSquares[i] += g_ti * g_ti;
                assert !Double.isNaN(gradSumSquares[i]);
                // Get the new learning rate.
                double h_tii = prm.constantAddend + Math.sqrt(gradSumSquares[i]);

                // Definitions
                int t0 = iterOfLastStep[i];
                int t = iterCount-1;
                double x_t0i = point.get(i);
                
                // Lazy update.
                double x_ti = FastMath.pow(h_t0ii / (prm.eta * prm.l2Lambda + h_t0ii), t - t0) * x_t0i;
                // Main update.                
                double x_t1i = (h_tii*x_ti - prm.eta*g_ti) / (prm.eta * prm.l2Lambda + h_tii);
                
                iterOfLastStep[i] = iterCount;
                assert !Double.isNaN(x_t1i);
                assert !Double.isInfinite(x_t1i);
                point.set(i, x_t1i);
                // Commented for speed.
                //log.debug(String.format("t=%d t0=%d i=%d g_ti=%.3g x_t0i=%.3g x_ti=%.3g x_t1i=%.3g", t, t0, i, g_ti, x_t0i, x_ti, x_t1i));
            }
        });
    }

    @Override
    protected double getEta0() {
        return prm.eta;
    }

    @Override
    protected void setEta0(double eta0) {
        prm.eta = eta0;
    }

}
