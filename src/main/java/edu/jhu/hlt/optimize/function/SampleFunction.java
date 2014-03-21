package edu.jhu.hlt.optimize.function;

import edu.jhu.hlt.optimize.BatchSampler;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * Wrapper of a DifferentiableBatchFunction which downselects to only a fixed
 * sample of the original examples.
 * 
 * @author mgormley
 */
public class SampleFunction extends AbstractDifferentiableBatchFunction implements DifferentiableBatchFunction {

    private DifferentiableBatchFunction function;
    private int[] sample;

    public SampleFunction(DifferentiableBatchFunction function, int sampleSize) {
        if (sampleSize > function.getNumExamples()) {
            throw new IllegalArgumentException("sampleSize must be <= the number of examples in the original batch function");
        }
        this.function = function;
        // Sample m indices from [1...n], where m = the sampleSize and n = the
        // number of examples in the original batch function.
        BatchSampler sampler = new BatchSampler(false, function.getNumExamples(), sampleSize);
        this.sample = sampler.sampleBatch();
        assert sample.length == sampleSize;
    }

    /**
     * Converts a batch indexing into the sample, to a batch indexing into the
     * original function.
     * 
     * @param batch The batch indexing into the sample.
     * @return A new batch indexing into the original function, containing only
     *         the indices from the sample.
     */
    private int[] convertBatch(int[] batch) {
        int[] conv = new int[batch.length];
        for (int i=0; i<batch.length; i++) {
            conv[i] = sample[batch[i]];
        }
        return conv;
    }

    @Override
    public double getValue(IntDoubleVector point, int[] batch) {
        return function.getValue(point, convertBatch(batch));
    }

    @Override
    public int getNumExamples() {
        return sample.length;
    }

    @Override
    public int getNumDimensions() {
        return function.getNumDimensions();
    }

    @Override
    public IntDoubleVector getGradient(IntDoubleVector point, int[] batch) {
        return function.getGradient(point, convertBatch(batch));
    }

    @Override
    public ValueGradient getValueGradient(IntDoubleVector point, int[] batch) {
        return function.getValueGradient(point, convertBatch(batch));
    }

}
