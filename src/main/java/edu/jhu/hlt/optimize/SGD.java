package edu.jhu.hlt.optimize;

import java.util.Date;

import org.apache.commons.lang3.mutable.MutableDouble;
import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.log4j.Logger;

import edu.jhu.hlt.optimize.BottouSchedule.BottouSchedulePrm;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.function.SampleFunction;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.hlt.util.OnOffLogger;
import edu.jhu.hlt.util.Prm;
import edu.jhu.prim.util.Lambda.FnIntDoubleToDouble;
import edu.jhu.prim.vector.IntDoubleVector;
import edu.jhu.util.Timer;

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
        /** The gain schedule which defines the learning rate at each iteration. */
        public GainSchedule sched = new BottouSchedule(new BottouSchedulePrm());
        /** Whether to automatically select the learning rate. (Leon Bottou's "secret ingredient".) */
        public boolean autoSelectLr = true;
        /** How many epochs between auto-select runs. */
        public int autoSelectFreq = 5;
        /** The number of passes over the dataset to perform. */
        public double numPasses = 10;
        /** The batch size to use at each step. */
        public int batchSize = 15;
        /** Whether batches should be sampled with replacement. */
        public boolean withReplacement = false;
        /** Date by which to stop. */
        public Date stopBy = null;
        /** Whether to compute the function value on the 0th iteration. */
        public boolean computeValueOnIterZero = true;
        public SGDPrm() { } 
        public SGDPrm(double initialLr, int numPasses, int batchSize) {
            this.sched.setEta0(initialLr);
            this.numPasses = numPasses;
            this.batchSize = batchSize;
        }
    }
    
    private static final OnOffLogger log = new OnOffLogger(Logger.getLogger(SGD.class));

    /** The number of gradient steps to run. */   
    private int iterations;
    /** The number of iterations performed thus far. */
    private int iterCount;
    /** The sampler of the indices for each batch. */
    private BatchSampler batchSampler;
   

    private SGDPrm prm;
    
    /**
     * Constructs an SGD optimizer.
     */
    public SGD(SGDPrm prm) {
        this.prm = prm;
    }
    
    /**
     * Initializes all the parameters for optimization.
     */
    protected void init(DifferentiableBatchFunction function) {
        int numExamples = function.getNumExamples();

        // Variables
        iterCount = 0;
        batchSampler = new BatchSampler(prm.withReplacement, numExamples, prm.batchSize);
                    
        // Constants
        iterations = (int) Math.ceil((double) prm.numPasses * numExamples / prm.batchSize);
        log.info("Setting number of batch gradient steps: " + iterations);
        
        prm.sched.init(function);
    }

    /**
     * Maximize the function starting at the given initial point.
     */
    @Override
    public boolean maximize(DifferentiableBatchFunction function, IntDoubleVector point) {
        return optimize(function, point, true);
    }

    /**
     * Minimize the function starting at the given initial point.
     */
    public boolean minimize(DifferentiableBatchFunction function, IntDoubleVector point) {
        return optimize(function, point, false);
    }

    private boolean optimize(DifferentiableBatchFunction function, final IntDoubleVector point, final boolean maximize) {
        init(function);
        optimizeWithoutInit(function, point, maximize);        
        // We don't test for convergence.
        return false;
    }

    private double optimizeWithoutInit(DifferentiableBatchFunction function, final IntDoubleVector point, final boolean maximize) {
        if (prm.stopBy != null) {
            log.debug("Max time alloted (hr): " + (prm.stopBy.getTime() - new Date().getTime()) / 1000. / 3600.);  
        }

        int passCount = 0;
        double passCountFrac = 0;

        if (prm.autoSelectLr) {
            autoSelectLr(function, point, maximize, prm);
        }
        
        double value = Double.NaN;
        if (prm.computeValueOnIterZero) {
            value = function.getValue(point);
            log.info(String.format("Function value on all examples = %g at iteration = %d on pass = %.2f", value, iterCount, passCountFrac));
        }
        assert (function.getNumDimensions() == point.getDimension());

        Timer timer = new Timer();
        timer.start();
        for (; iterCount < iterations; iterCount++) {
            int[] batch = batchSampler.sampleBatch();
            
            // Get the current value and gradient of the function.
            ValueGradient vg = function.getValueGradient(point, batch);
            value = vg.getValue();
            final IntDoubleVector gradient = vg.getGradient();
            log.trace(String.format("Function value on batch = %g at iteration = %d", value, iterCount));
            prm.sched.takeNoteOfGradient(gradient);
            
            // Scale the gradient by the parameter-specific learning rate.
            gradient.apply(new FnIntDoubleToDouble() {
                @Override
                public double call(int index, double value) {
                    double lr = prm.sched.getLearningRate(iterCount, index);
                    if (maximize) {
                        return lr * value;
                    } else {
                        return - lr * value;
                    }
                }
            });
            
            // Take a step in the direction of the gradient.
            point.add(gradient);

            // If a full pass through the data has been completed...
            passCountFrac = (double) iterCount * prm.batchSize / function.getNumExamples();
            if ((int) Math.floor(passCountFrac) > passCount || iterCount == iterations - 1) {
                // Compute the average learning rate and the average step size.
                final MutableDouble avgLr = new MutableDouble(0.0);
                final MutableDouble avgStep = new MutableDouble(0d);
                final MutableInt numNonZeros = new MutableInt(0);
                gradient.apply(new FnIntDoubleToDouble() {
                    @Override
                    public double call(int index, double value) {
                        double lr = prm.sched.getLearningRate(iterCount, index);
                        assert !Double.isNaN(point.get(index));
                        if (value != 0.0) {
                            avgLr.add(lr);
                            avgStep.add(gradient.get(index));
                            numNonZeros.increment();
                        }
                        return value;
                    }
                });
                avgLr.setValue(avgLr.doubleValue() / numNonZeros.doubleValue());
                avgStep.setValue(avgStep.doubleValue() / numNonZeros.doubleValue());
                log.debug("Average learning rate: " + avgLr);
                log.debug("Average step size: " + avgStep);
                
                // Another full pass through the data has been completed or we're on the last iteration.
                // Get the value of the function on all the examples.
                value = function.getValue(point);
                log.info(String.format("Function value on all examples = %g at iteration = %d on pass = %.2f", value, iterCount, passCountFrac));
                log.debug(String.format("Average time per pass (min): %.2g", timer.totSec() / 60.0 / passCountFrac));
            }
            if ((int) Math.floor(passCountFrac) > passCount) {
                // Another full pass through the data has been completed.
                passCount++;

                if (prm.autoSelectLr && (passCount % prm.autoSelectFreq == 0)) {
                    // Auto select every autoSelecFreq epochs.
                    autoSelectLr(function, point, maximize, prm);
                }
            }
            
            if (prm.stopBy != null) {
                Date now = new Date();
                if (now.after(prm.stopBy)) {
                    log.info(String.format("Current time is after stop-by time. now=%s, stopBy=%s", now.toString(), prm.stopBy.toString()));
                    log.info("Stopping training early.");
                    break;
                }
            }
        }
        
        return value;
    }

    protected void autoSelectLr(DifferentiableBatchFunction function, final IntDoubleVector point, final boolean maximize, SGDPrm origPrm) {
        double eta0 = autoSelectLrStatic(function, point, maximize, origPrm, iterCount);
        prm.sched.setEta0(eta0);
    }
    
    private static double autoSelectLrStatic(DifferentiableBatchFunction function, final IntDoubleVector point, final boolean maximize, SGDPrm origPrm, int iterCount) {
        log.info("Auto-selecting the best learning rate constant");
        // Parameters for how we perform auto selection of the intial learning rate.        
        double factor = 2;
        int numEvals = 10;
        // This sample size equates to a single epoch.
        int sampleSize = (int) Math.ceil((double) function.getNumExamples() / numEvals);
        SampleFunction sampFunction = new SampleFunction(function, sampleSize); 
        //
        // Get the objective value with no training.
        double startObj = sampFunction.getValue(point);
        log.info("Initial sample obj="+startObj);
        // Initialize the "best" values.
        double origEta0 = origPrm.sched.getEta0();
        double bestEta = origEta0;
        double bestObj = startObj;
        
        boolean increasing = true;
        double eta = origEta0;
        for (int i=0; i<numEvals; i++) {
            double obj = evaluateInitialLr(sampFunction, point, maximize, origPrm, eta, iterCount);
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

    private static double evaluateInitialLr(DifferentiableBatchFunction sampFunction, IntDoubleVector origPoint, boolean maximize, SGDPrm origPrm, double eta, int iterCount) {
        SGDPrm prm = Prm.clonePrm(origPrm);
        IntDoubleVector point = origPoint.copy();
        prm.sched = prm.sched.copy();
        prm.sched.setEta0(eta);
        prm.numPasses = 1; // Only one epoch.
        prm.autoSelectLr = false; // Don't recurse.
        prm.computeValueOnIterZero = false;
        
        SGD sgd = new SGD(prm);
        log.setEnabled(false);
        sgd.init(sampFunction);
        // Make sure we start off the learning rate schedule at the proper place.
        sgd.iterCount += iterCount;
        sgd.iterations += iterCount;
        double obj = sgd.optimizeWithoutInit(sampFunction, point, maximize);
        log.setEnabled(true);
        return obj;
    }

    private static boolean isBetter(double obj, double bestObj, boolean maximize) {
        return maximize ? obj > bestObj : obj < bestObj;
    }
    
}
