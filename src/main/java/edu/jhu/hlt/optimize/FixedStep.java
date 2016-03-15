package edu.jhu.hlt.optimize;

/**
 * Fixed step
 */
public class FixedStep extends  EmptyGainSchedule {

    private static final long serialVersionUID = 1L;
    private double learningRate;

    public FixedStep(double learningRate) {
        this.learningRate = learningRate;
    }
    
    @Override
    public double getLearningRate(int iterCount, int i) {
        return learningRate;
    }

    @Override
    public boolean isSameForAllParameters() {
        return true;
    }

}
