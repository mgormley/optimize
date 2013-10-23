package edu.jhu.hlt.optimize;

public interface Proposable {

    Proposer getProposer();
    double[] samplePoint();

}
