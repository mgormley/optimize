package edu.jhu.hlt.optimize;

import java.util.Date;

import org.apache.commons.lang3.mutable.MutableDouble;
import org.apache.commons.lang3.mutable.MutableInt;
import org.slf4j.LoggerFactory;

import edu.jhu.hlt.optimize.BottouSchedule.BottouSchedulePrm;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.function.Function;
import edu.jhu.hlt.optimize.function.NonstationaryFunction;
import edu.jhu.hlt.optimize.function.SampleFunction;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.hlt.util.OnOffLogger;
import edu.jhu.hlt.util.Prm;
import edu.jhu.prim.util.Lambda.FnIntDoubleToDouble;
import edu.jhu.prim.util.Timer;
import edu.jhu.prim.vector.IntDoubleDenseVector;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * Stochastic gradient descent with minibatches.
 * 
 * We use the gain schedule suggested in Leon Bottou's (2012) SGD Tricks paper.
 * 
 * @author mgormley
 */
public class SGD implements Optimizer<DifferentiableBatchFunction> {

    /** Options for this optimizer. */
    public static class SGDPrm extends Prm {
        private static final long serialVersionUID = 1L;
        /** The gain schedule which defines the learning rate at each iteration. */
        public GainSchedule sched = new BottouSchedule(new BottouSchedulePrm());
        /** Whether to automatically select the learning rate. (Leon Bottou's "secret ingredient".) */
        public boolean autoSelectLr = true;
        /** How many epochs between auto-select runs. */
        public int autoSelectFreq = 5;
        /** The number of passes over the dataset to perform. */
        public int numPasses = 10;
        /** The batch size to use at each step. */
        public int batchSize = 15;
        /** Whether batches should be sampled with replacement. */
        public boolean withReplacement = false;
        /** Date by which to stop. */
        public Date stopBy = null;
        /** Whether to compute the function value on the non-final iterations. */
        public boolean computeValueOnNonFinalIter = true;
        /** Whether to return the point with min validation score. */
        public boolean earlyStopping = true;
        /** Whether to do parameter averaging. */
        public boolean averaging = false;
        /** The pass at which to begin averaging of the parameters. */
        public double passToStartAvg = 1.0;
        public SGDPrm() { }
    }
    
    private static final OnOffLogger log = new OnOffLogger(LoggerFactory.getLogger(SGD.class));

    private SGDPrm prm;
    
    /**
     * Constructs an SGD optimizer.
     */
    public SGD(SGDPrm prm) {
        this.prm = prm;
    }

    protected SGD copy() {
        // Don't copy the bestPoint / bestDevScore.
        return new SGD(Prm.clonePrm(this.prm));
    }
    
    /**
     * Initializes all the parameters for optimization.
     */
    protected void init(DifferentiableBatchFunction function, IntDoubleVector point) {
        prm.sched.init(function);
    }

    /**
     * Maximize the function starting at the given initial point.
     */
    @Override
    public boolean maximize(DifferentiableBatchFunction function, IntDoubleVector point) {
        return optimize(function, point, true, null);
    }

    /**
     * Minimize the function starting at the given initial point.
     */
    public boolean minimize(DifferentiableBatchFunction function, IntDoubleVector point) {
        return optimize(function, point, false, null);
    }

    public boolean optimize(DifferentiableBatchFunction function, final IntDoubleVector point, 
            final boolean maximize, Function devLoss) {
        init(function, point);
        final int itersPerEpoch = getItersPerPass(function);
        log.info("Number of batch gradient iterations: " + prm.numPasses * itersPerEpoch);
        optimizeWithoutInit(function, point, maximize, itersPerEpoch, 0, devLoss);
        // We don't test for convergence.
        return false;
    }

    /** Gets the number of (batch) iterations per epoch (i.e. pass through the training data). */
    protected int getItersPerPass(DifferentiableBatchFunction function) {
        return (int) Math.ceil((double) function.getNumExamples() / prm.batchSize);
    }

    private double optimizeWithoutInit(DifferentiableBatchFunction function, final IntDoubleVector point, 
            final boolean maximize, final int itersPerPass, int pass, Function devLoss) {
        final int maxIters = prm.numPasses * itersPerPass;
        final int startIter = pass * itersPerPass;
        int iter = startIter;
        BatchSampler batchSampler = new BatchSampler(prm.withReplacement, function.getNumExamples(), prm.batchSize);
        Timer passTimer = new Timer();
        Timer tuneTimer = new Timer();
        double bestDevLoss = Double.MAX_VALUE;
        IntDoubleVector bestPoint = prm.earlyStopping && devLoss != null ? new IntDoubleDenseVector(function.getNumDimensions()) : null;
        IntDoubleVector avgPoint = prm.averaging ? new IntDoubleDenseVector(point) : null;
        assert !prm.averaging || prm.passToStartAvg >= pass;
        
        // Setup.
        if (prm.stopBy != null) {
            log.debug("Max time alloted (hr): " + (prm.stopBy.getTime() - new Date().getTime()) / 1000. / 3600.);  
        }
        if (function instanceof NonstationaryFunction) {
            ((NonstationaryFunction) function).updatateIterAndMax(iter, maxIters);
        }
        assert (function.getNumDimensions() >= point.getNumImplicitEntries());

        // Optimization.
        passTimer.start();
        passLoop:
        for (; pass < prm.numPasses; pass++) {
            if (prm.autoSelectLr && (pass % prm.autoSelectFreq == 0)) {
                passTimer.stop();
                tuneTimer.start();
                // Auto select every autoSelecFreq epochs.
                autoSelectLr(function, point, maximize, pass);
                log.info("Average time (min) per tuning pass: " + tuneTimer.avgSec() / 60.0);
                tuneTimer.stop();
                passTimer.start();
            }
            if (prm.computeValueOnNonFinalIter) {
                bestDevLoss = sufferLossAndUpdateBest(function, point, avgPoint, pass, devLoss, startIter,
                        iter, bestDevLoss, bestPoint)[1];
            }
            
            // Make a full pass through the training data.
            for (int i=0; i<itersPerPass; i++) {
                int[] batch = batchSampler.sampleBatch();
                
                if (function instanceof NonstationaryFunction) {
                    ((NonstationaryFunction) function).updatateIterAndMax(iter, maxIters);
                }
                
                // Get the current value and gradient of the function.
                ValueGradient vg = function.getValueGradient(point, batch);
                double value = vg.getValue();
                final IntDoubleVector gradient = vg.getGradient();
                log.trace(String.format("Function value on batch = %g at iteration = %d", value, iter));
                prm.sched.takeNoteOfGradient(gradient);
                
                // Step in the direction of the gradient (maximization) or opposite it (minimization).
                takeGradientStep(point, gradient, maximize, iter);                
                logAvgLrAndStepSize(point, gradient, iter);
                logStatsAboutPoint(point);
                
                if (prm.averaging) {
                    // Non-sparse update of averaged parameters.
                    double t0 = prm.passToStartAvg * itersPerPass;
                    int t = iter + 1;
                    double mu_t = 1.0 / Math.max(1, t - t0); 
                    for (int m=0; m<function.getNumDimensions(); m++) {
                        avgPoint.set(m, (1 - mu_t) * avgPoint.get(m) + mu_t * point.get(m));
                    }
                }
                if (prm.stopBy != null) {
                    Date now = new Date();
                    if (now.after(prm.stopBy)) {
                        log.info(String.format("Current time is after stop-by time. now=%s, stopBy=%s", now.toString(), prm.stopBy.toString()));
                        log.info("Stopping training early.");                        
                        break passLoop;
                    }
                }
                iter++;
            }
            // Another full pass through the data has been completed.
            log.debug(String.format("Average time per pass (min): %.2g", passTimer.totSec() / 60.0 / (pass + 1)));
        }
        
        double[] pair = sufferLossAndUpdateBest(function, point, avgPoint, pass, devLoss, startIter, 
                iter, bestDevLoss, bestPoint);
        double value = pair[0];
        bestDevLoss = pair[1];
        if (prm.earlyStopping && devLoss != null) {
            // Return the best point seen so far.
            log.debug("Early stopping returning point with dev loss: " + bestDevLoss);
            for (int m=0; m<function.getNumDimensions(); m++) {
                point.set(m, bestPoint.get(m));
            }
        } else if (prm.averaging) {
            log.debug("Returning averaged parameters.");
            for (int m=0; m<function.getNumDimensions(); m++) {
                point.set(m, avgPoint.get(m));
            }
        }
        return value;
    }

    protected double[] sufferLossAndUpdateBest(DifferentiableBatchFunction function, IntDoubleVector point,
            IntDoubleVector avgPoint, int pass, Function devLoss, int startIter, int iter, double bestDevLoss, 
            IntDoubleVector bestPoint) {
        if (prm.averaging) {
            point = avgPoint;
        }
        
        // Report the value of the function on all the examples.
        double value = function.getValue(point);
        log.info(String.format("Function value on all examples = %g at iteration = %d on pass = %d", value, iter, pass));
        
        if (devLoss != null) {
            // Report the loss on validation data.
            double devScore = devLoss.getValue(point);
            log.info(String.format("Dev loss = %g at iteration = %d on pass = %d", devScore, iter, pass));
            if (prm.earlyStopping && devScore < bestDevLoss) {
                // Store the best point seen so far.
                for (int m=0; m<function.getNumDimensions(); m++) {
                    bestPoint.set(m, point.get(m));
                }
                bestDevLoss = devScore;
            }
        }
        return new double[]{value, bestDevLoss};
    }

    protected void takeGradientStep(final IntDoubleVector point, final IntDoubleVector gradient, 
            final boolean maximize, final int iterCount) {
        // Scale the gradient by the parameter-specific learning rate.
        gradient.apply(new FnIntDoubleToDouble() {
            @Override
            public double call(int index, double value) {
                double lr = prm.sched.getLearningRate(iterCount, index);
                if (maximize) {
                    value = lr * value;
                } else {
                    value = - lr * value;
                }
                assert !Double.isNaN(value);
                assert !Double.isInfinite(value);
                return value;
            }
        });
        
        // Take a step in the direction of the gradient.
        point.add(gradient);
    }

    private void logAvgLrAndStepSize(final IntDoubleVector point, final IntDoubleVector gradient, final int iterCount) {
        if (log.isTraceEnabled()) {
            // Compute the average learning rate and the average step size.
            final MutableDouble avgLr = new MutableDouble(0.0);
            final MutableDouble grad2norm = new MutableDouble(0d);
            final MutableInt numNonZeros = new MutableInt(0);
            gradient.apply(new FnIntDoubleToDouble() {
                @Override
                public double call(int index, double value) {
                    double lr = prm.sched.getLearningRate(iterCount, index);
                    assert !Double.isNaN(point.get(index));
                    if (value != 0.0) {
                        avgLr.add(lr);
                        double grad_i = gradient.get(index);
                        grad2norm.add(grad_i * grad_i);
                        numNonZeros.increment();
                    }
                    return value;
                }
            });
            avgLr.setValue(avgLr.doubleValue() / numNonZeros.doubleValue());
            grad2norm.setValue(Math.sqrt(grad2norm.doubleValue()));
            if (numNonZeros.doubleValue() == 0) {
                avgLr.setValue(0.0);
                grad2norm.setValue(0.0);
            }
            log.trace("Average learning rate: " + avgLr);
            log.trace("Step 2-norm: " + grad2norm);
        }
    }

    private void logStatsAboutPoint(IntDoubleVector point) {
        if (log.isTraceEnabled()) {
            log.trace(String.format("min=%g max=%g infnorm=%g l2=%g", point.getMin(), point.getMax(), point.getInfNorm(), point.getL2Norm()));
        }
    }

    protected void autoSelectLr(DifferentiableBatchFunction function, final IntDoubleVector point, 
            final boolean maximize, final int pass) {
        double eta0 = autoSelectLrStatic(function, point, maximize, this, pass);
        this.setEta0(eta0);
    }
    
    private static double autoSelectLrStatic(DifferentiableBatchFunction function, final IntDoubleVector point, 
            final boolean maximize, SGD orig, int pass) {
        log.info("Auto-selecting the best learning rate constant at pass " + pass);
        // Parameters for how we perform auto selection of the initial learning rate.
        // The max number of iterations.
        int numEvals = 10;
        // How to update the learning rate at each iteration (e.g. 2 yeilds doubling, then halving)
        double factor = 2;
        // This sample size equates to a single epoch.
        int sampleSize = (int) Math.ceil((double) function.getNumExamples() / numEvals);
        SampleFunction sampFunction = new SampleFunction(function, sampleSize); 
        //
        // Get the objective value with no training.
        double startObj = sampFunction.getValue(point);
        log.info("Initial sample obj="+startObj);
        // Initialize the "best" values.
        double origEta0 = orig.getEta0();
        double bestEta = origEta0;
        double bestObj = startObj;
        
        boolean increasing = true;
        double eta = origEta0;
        for (int i=0; i<numEvals; i++) {
            double obj = evaluateInitialLr(sampFunction, point, maximize, orig, eta, pass);
            log.info(String.format("Evaluated initial learning rate: eta="+eta+" obj="+obj));
            if (isBetter(obj, bestObj, maximize)) {
                bestObj = obj;
                bestEta = eta;
            }
            if (!isBetter(obj, startObj, maximize) && increasing) {
                // If training caused the objective to worsen, then switch from
                // increasing the learning rate to decreasing it.
                increasing = false;
                eta = origEta0;
            }
            if (increasing) {
                // Increase eta by a factor.
                eta *= factor;
            } else {
                // Decrease eta by a factor.
                eta /= factor;
            }
        }
        // Conservatively return the value for eta smaller than the best one.
        bestEta = bestEta / factor;
        log.info("Chose initial learning rate: eta="+bestEta);
        
        return bestEta;
    }

    private static double evaluateInitialLr(DifferentiableBatchFunction sampFunction, IntDoubleVector origPoint, 
            boolean maximize, SGD orig, double eta, int pass) {
        SGD sgd = orig.copy();
        if (orig.prm == sgd.prm) { throw new IllegalStateException("Copy must create a new prm."); }
        IntDoubleVector point = origPoint.copy();
        SGDPrm prm = sgd.prm;
        prm.sched = prm.sched.copy();
        sgd.setEta0(eta);
        prm.numPasses = 1; // Only one epoch.
        prm.autoSelectLr = false; // Don't recurse.
        prm.computeValueOnNonFinalIter = false; // Report function value only at end.
        prm.earlyStopping = false;
        prm.averaging = false;
        
        log.setEnabled(false);
        sgd.init(sampFunction, point);
        // Make sure we start off the learning rate schedule at the proper place.
        final int itersPerPass = sgd.getItersPerPass(sampFunction);
        double obj;
        try {
            obj = sgd.optimizeWithoutInit(sampFunction, point, maximize, itersPerPass, pass, null);
        } catch (Throwable t) {
            log.setEnabled(true);
            String msg = (t.getMessage() == null) ? "": " : " + t.getMessage();
            log.error("Failed to evaluate hyperparameter. Caught throwable: " + t.getClass() + msg);
            log.trace("Stacktrace from previous ERROR:\n", t);
            obj = worstObjValue(maximize);
        }
        log.setEnabled(true);
        return obj;
    }

    protected double getEta0() {
        return prm.sched.getEta0();
    }

    protected void setEta0(double eta) {
        prm.sched.setEta0(eta);
    }

    private static boolean isBetter(double obj, double bestObj, boolean maximize) {
        return maximize ? obj > bestObj : obj < bestObj;
    }
    
    private static double worstObjValue(boolean maximize) {
        return maximize ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
    }
    
}
