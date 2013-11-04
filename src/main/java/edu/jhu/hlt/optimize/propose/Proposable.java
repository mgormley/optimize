package edu.jhu.hlt.optimize.propose;

public interface Proposable {

    Proposer getProposer();
    double[] samplePoint();

}
