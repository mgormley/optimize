package edu.jhu.hlt.optimize;

import java.util.Arrays;

import org.apache.log4j.Logger;

import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.util.Prm;
import edu.jhu.prim.arrays.DoubleArrays;
import edu.jhu.prim.arrays.IntArrays;
import edu.jhu.prim.util.Lambda.FnIntDoubleToVoid;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * AdaGrad with a Composite Mirror Descent update and L1 regularizer with lazy updates (Duchi et al., 2011).
 * 
 * @author mgormley
 */
public class AdaGradComidL1 extends SGD implements Optimizer<DifferentiableBatchFunction> {

    /** Options for this optimizer. */
    public static class AdaGradComidL1Prm extends SGDPrm {
        private static final long serialVersionUID = 1L;
        /** The scaling parameter for the learning rate. */
        public double eta = 0.1;
        /**
         * The amount added (delta) to the sum of squares outside the square
         * root. This is to combat the issue of tiny gradients throwing the hole
         * optimization off early on.
         */
        public double constantAddend = 1e-9;
        /** The weight on the l1 regularizer. */
        public double l1Lambda = 0.0;
    }

    private static final long serialVersionUID = 1L;
    private static final Logger log = Logger.getLogger(AdaGradComidL1.class);

    private final AdaGradComidL1Prm prm;
    // The iteration of the last step taken for each model parameter.
    private int[] iterOfLastStep;
    // Sum of squares of gradients up to current time for each parameter.
    private double[] gradSumSquares;
    
    /**
     * Constructs an SGD optimizer.
     */
    public AdaGradComidL1(AdaGradComidL1Prm prm) {
        super(prm);
        if (!(prm.sched == null || prm.sched instanceof EmptyGainSchedule)) {
            throw new IllegalArgumentException("Schedule for AdaGrad must be null.");
        }
        this.prm = prm;
        this.prm.sched = new EmptyGainSchedule();
    }
    
    @Override
    public AdaGradComidL1 copy() {
        AdaGradComidL1 sgd = new AdaGradComidL1(Prm.clonePrm(this.prm));
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
    }

    @Override
    protected void takeGradientStep(final IntDoubleVector point, final IntDoubleVector gradient, 
            final boolean maximize, final int iterCount) {
        gradient.iterate(new FnIntDoubleToVoid() {
            @Override
            public void call(int i, double g_ti) {
                g_ti = maximize ? -g_ti : g_ti;
                // Get the old learning rate.
                double lr_t0 = getLearningRate(i);
                // Update the sum of squares.
                gradSumSquares[i] += g_ti * g_ti;
                assert !Double.isNaN(gradSumSquares[i]);
                // Get the new learning rate.
                double lr_t = getLearningRate(i);

                // Definitions
                int t0 = iterOfLastStep[i];
                int t = iterCount-1;
                double x_t0i = point.get(i);
                
                // Lazy update.
                double x_ti = (x_t0i < 0 ? -1 : 1) * Math.max(0, Math.abs(x_t0i) - prm.l1Lambda * lr_t0 * (t - t0));
                // Main update.                
                double xtimlrg = x_ti - lr_t * g_ti;
                double x_t1i = (xtimlrg < 0 ? -1 : 1) * Math.max(0,  Math.abs(xtimlrg) - prm.l1Lambda * lr_t);
                
                iterOfLastStep[i] = iterCount;
                assert !Double.isNaN(x_t1i);
                assert !Double.isInfinite(x_t1i);
                point.set(i, x_t1i);
                // Commented for speed.
                //log.debug(String.format("t=%d t0=%d i=%d g_ti=%.3g x_t0i=%.3g x_ti=%.3g x_t1i=%.3g", t, t0, i, g_ti, x_t0i, x_ti, x_t1i));
            }
        });
    }
    
    /**
     * Gets the learning rate for the current iteration.
     * @param i The index of the current model parameter. 
     */
    private double getLearningRate(int i) {
        if (gradSumSquares[i] < 0) {
            throw new RuntimeException("Gradient sum of squares entry is < 0: " + gradSumSquares[i]);
        }
        double learningRate = prm.eta / (prm.constantAddend + Math.sqrt(gradSumSquares[i]));
        assert !Double.isNaN(learningRate);
        if (learningRate == Double.POSITIVE_INFINITY) {
            if (gradSumSquares[i] != 0.0) {
                log.warn("Gradient was non-zero but learningRate hit positive infinity: " + gradSumSquares[i]);
            }
            // Just return zero. The gradient is probably 0.0.
            return 0.0;
        }
        return learningRate;
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
