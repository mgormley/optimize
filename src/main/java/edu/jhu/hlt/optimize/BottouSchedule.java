package edu.jhu.hlt.optimize;

import java.util.Date;

import org.apache.commons.lang3.mutable.MutableDouble;
import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.log4j.Logger;

import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.function.SampleFunction;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.hlt.util.OnOffLogger;
import edu.jhu.hlt.util.Prm;
import edu.jhu.prim.util.Lambda.FnIntDoubleToDouble;
import edu.jhu.prim.vector.IntDoubleVector;
import edu.jhu.util.Timer;

/**
 * The gain schedule suggested in Leon Bottou's (2012) SGD Tricks paper.
 * 
 * @author mgormley
 */
public class BottouSchedule implements GainSchedule {

    /** Options for this class. */
    public static class BottouSchedulePrm extends Prm {
        /**
         * The initial learning rate. (i.e. \gamma_0 in where \gamma_t =
         * \frac{\gamma_0}{1 + \gamma_0 \lambda t})
         */
        public double initialLr = 0.1;
        /**
         * Learning rate scaler. (i.e. \lambda in where \gamma_t =
         * \frac{\gamma_0}{1 + \gamma_0 \lambda t})
         * 
         * According to Leon Bottou's (2012) SGD tricks paper, when using an L2
         * regularizer of the form \frac{\lambda}{2} ||w||^2, where w is the
         * weight vector, this should be set to the value \lambda. If the L2
         * regularizer is instead parameterized by the variance of the L2 (i.e.
         * Guassian) prior, then we should set \lambda = 1 / \sigma^2.
         */
        public double lambda = 1.0;
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
        // \gamma_t = \frac{\gamma_0}{1 + \gamma_0 \lambda t})
        //
        return prm.initialLr / (1 + prm.initialLr * prm.lambda * iterCount);
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
