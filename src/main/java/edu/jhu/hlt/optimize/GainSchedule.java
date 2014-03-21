package edu.jhu.hlt.optimize;

import java.io.Serializable;

import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.prim.vector.IntDoubleVector;

public interface GainSchedule extends Serializable {

    void init(DifferentiableBatchFunction function);
    void takeNoteOfGradient(IntDoubleVector gradient);
    double getLearningRate(int iterCount, int i);
    
    GainSchedule copy();
    double getEta0();
    void setEta0(double eta0);
        
}
