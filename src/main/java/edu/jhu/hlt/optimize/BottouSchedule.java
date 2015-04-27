package edu.jhu.hlt.optimize;

import java.util.Date;

import org.apache.commons.lang3.mutable.MutableDouble;
import org.apache.commons.lang3.mutable.MutableInt;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.function.SampleFunction;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.hlt.util.OnOffLogger;
import edu.jhu.hlt.util.Prm;
import edu.jhu.prim.util.Timer;
import edu.jhu.prim.util.Lambda.FnIntDoubleToDouble;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * The gain schedule suggested in Leon Bottou's (2012) SGD Tricks paper.
 * 
 * \gamma_t = \frac{\gamma_0}{(1 + \gamma_0 \lambda t)^p}
 * 
 * @author mgormley
 */
public class BottouSchedule implements GainSchedule {

    /** Options for this class. */
    public static class BottouSchedulePrm extends Prm {
        /**
         * The initial learning rate. (i.e. \gamma_0)
         */
        public double initialLr = 0.1;
        /**
         * Learning rate scaler. (i.e. \lambda)
         * 
         * According to Leon Bottou's (2012) SGD tricks paper, when using an L2
         * regularizer of the form \frac{\lambda}{2} ||w||^2, where w is the
         * weight vector, this should be set to the value \lambda. If the L2
         * regularizer is instead parameterized by the variance of the L2 (i.e.
         * Guassian) prior, then we should set \lambda = 1 / \sigma^2.
         */
        public double lambda = 1.0;
        /**
         * The power to raise the denominator of the learning rate. (i.e. p)
         * For SGD p = 1.0, for ASGD p = 0.75.
         */
        public double power = 1.0;
    }
    
    private BottouSchedulePrm prm;
    
    public BottouSchedule(BottouSchedulePrm prm) {
        this.prm = prm;
    }

    /**
     * Gets the learning rate for the current iteration.
     * @param iterCount The current iteration.
     * @param i The index of the current model parameter. 
     */
    @Override
    public double getLearningRate(int iterCount, int i) {
        // We use the learning rate suggested in Leon Bottou's (2012) SGD Tricks paper.
        // 
        // \gamma_t = \frac{\gamma_0}{(1 + \gamma_0 \lambda t)^p}
        //
        // For SGD p = 1.0, for ASGD p = 0.75
        if (prm.power == 1.0) {
            return prm.initialLr / (1 + prm.initialLr * prm.lambda * iterCount);
        } else {
            return prm.initialLr / Math.pow(1 + prm.initialLr * prm.lambda * iterCount, prm.power);
        }
    }
    
    @Override
    public void init(DifferentiableBatchFunction function) { 
        // Do nothing.
    }

    @Override
    public void takeNoteOfGradient(IntDoubleVector gradient) {
        // Do nothing.
    }

    @Override
    public GainSchedule copy() {
        BottouSchedulePrm otherPrm = Prm.clonePrm(this.prm);
        BottouSchedule other = new BottouSchedule(otherPrm);
        return other;
    }

    @Override
    public double getEta0() {
        return prm.initialLr;
    }

    @Override
    public void setEta0(double eta0) {
        prm.initialLr = eta0;        
    }

    @Override
    public boolean isSameForAllParameters() {
        return true;
    }

}
