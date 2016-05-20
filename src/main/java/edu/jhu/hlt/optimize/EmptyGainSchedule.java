package edu.jhu.hlt.optimize;

import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * Fake gain schedule.
 * @author mgormley
 */
class EmptyGainSchedule implements  GainSchedule {

    private static final long serialVersionUID = 1L;
    
    /**
     * Constructs an SGD optimizer.
     */
    public EmptyGainSchedule() { }
    
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
        throw new RuntimeException("no-op");
    }

    @Override
    public void setEta0(double eta0) {
        throw new RuntimeException("no-op");
    }
    
    @Override
    public boolean isSameForAllParameters() {
        return false;
    }

    @Override
    public GainSchedule copy() {
        return this;
    }

}
