package edu.jhu.hlt.optimize;

import java.util.Arrays;

import org.apache.log4j.Logger;

import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.util.Prm;
import edu.jhu.prim.arrays.DoubleArrays;
import edu.jhu.prim.arrays.IntArrays;
import edu.jhu.prim.list.DoubleArrayList;
import edu.jhu.prim.util.Lambda.FnIntDoubleToVoid;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * AdaGrad with a Composite Mirror Descent update and L1 regularizer with lazy updates (Duchi et al., 2011).
 * 
 * @author mgormley
 */
public class AdaGradComidL1 extends SGD implements Optimizer<DifferentiableBatchFunction>, GainSchedule {

    /** Options for this optimizer. */
    public static class AdaGradFobosPrm extends SGDPrm {
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

    private final AdaGradFobosPrm prm;
    // The iteration of the last step taken for each model parameter.
    private int[] iterOfLastStep;
    // Sum of squares of gradients up to current time for each parameter.
    private double[] gradSumSquares;
    // Last value of model parameter before update. 
    private double[] prevParams;
    
    /**
     * Constructs an SGD optimizer.
     */
    public AdaGradComidL1(AdaGradFobosPrm prm) {
        super(prm);
        if (prm.sched != null) {
            throw new IllegalArgumentException("Schedule for AdaGrad must be null.");
        }
        this.prm = prm;
        this.prm.sched = this;
    }
    
    @Override
    public AdaGradComidL1 copy() {
        AdaGradComidL1 sgd = new AdaGradComidL1(Prm.clonePrm(this.prm));
        sgd.iterOfLastStep = IntArrays.copyOf(this.iterOfLastStep);
        sgd.gradSumSquares = DoubleArrays.copyOf(this.gradSumSquares);
        sgd.prevParams = DoubleArrays.copyOf(this.prevParams);
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
        this.prevParams = new double[function.getNumDimensions()];
        for (int i=0; i<prevParams.length; i++) {
            prevParams[i] = point.get(i);
        }
    }

    @Override
    protected void takeGradientStep(final IntDoubleVector point, final IntDoubleVector gradient, 
            final boolean maximize, final int iterCount) {
        gradient.iterate(new FnIntDoubleToVoid() {
            @Override
            public void call(int i, double gr) {
                gr = maximize ? -gr : gr;
                // Get the old learning rate.
                double lrt0 = getLearningRate(i);
                // Update the sum of squares.
                gradSumSquares[i] += gr * gr;
                assert !Double.isNaN(gradSumSquares[i]);
                // Get the new learning rate.
                double lrt = getLearningRate(i);

                // Definitions
                int t0 = iterOfLastStep[i];
                int t = iterCount-1;
                double xt0i = point.get(i);
                
                // Lazy update.
                double xti = (xt0i < 0 ? -1 : 1) * Math.max(0, Math.abs(xt0i) - prm.l1Lambda * lrt0 * (t - t0));
                // Main update.                
                double xtimlrg = xti - lrt * gr;
                double xt1i = (xtimlrg < 0 ? -1 : 1) * Math.max(0,  Math.abs(xtimlrg) - prm.l1Lambda * lrt);
                
                iterOfLastStep[i] = iterCount;
                assert !Double.isNaN(xt1i);
                assert !Double.isInfinite(xt1i);
                prevParams[i] = point.set(i, xt1i);
                log.debug(String.format("t=%d t0=%d i=%d gr=%.3g xt0i=%.3g xti=%.3g xt1i=%.3g", t, t0, i, gr, xt0i, xti, xt1i));
            }
        });
    }
    
    /**
     * Gets the learning rate for the current iteration.
     * @param i The index of the current model parameter. 
     */
    public double getLearningRate(int i) {
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
    public void init(DifferentiableBatchFunction function) {
        // Do nothing to intialize the schedule.
    }
    
    @Override
    public void takeNoteOfGradient(IntDoubleVector gradient) {
        // Do nothing. We update the gradient sum of squares in takeGradientStep().
    }

    @Override
    public double getLearningRate(int iterCount, int i) {
        throw new RuntimeException("This method should never be called");
    }

    @Override
    public double getEta0() {
        return prm.eta;
    }

    @Override
    public void setEta0(double eta0) {
        prm.eta = eta0;
    }
    
    @Override
    public boolean isSameForAllParameters() {
        return false;
    }

}
