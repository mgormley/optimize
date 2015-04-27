package edu.jhu.hlt.optimize;

import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.prim.arrays.IntArrays;
import edu.jhu.prim.sort.IntSort;
import edu.jhu.prim.util.Prng;

/**
 * Sampler of batches.
 * 
 * @author mgormley
 */
public class BatchSampler {

    private static final Logger log = LoggerFactory.getLogger(BatchSampler.class);

    // Parameters.
    private boolean withReplacement;
    private int numExamples;
    private int batchSize;
    
    // Cache of (shuffled) indices for sampling without replacement.
    private int[] indices;
    // Current index to sample from next.
    private int cur;

    public BatchSampler(boolean withReplacement, int numExamples, int batchSize) {
        this.withReplacement = withReplacement;
        this.numExamples = numExamples;
        this.batchSize = batchSize;
        
        if (!withReplacement) {
            indices = IntSort.getIndexArray(numExamples);
            cur = 0;
        }
    }
    
    public int[] sampleBatch() {
        final int[] batch;
        if (withReplacement) {
            batch = sampleBatchWithReplacement();
        } else {
            batch = sampleBatchWithoutReplacement();
        }
        if (log.isTraceEnabled()) {
            log.trace("Sampled batch: " + Arrays.toString(batch));
        }
        return batch;
    }

    /** Samples a batch of indices in the range [0, numExamples) with replacement. */
    public int[] sampleBatchWithReplacement() {
        // Sample the indices with replacement.
        int[] batch = new int[batchSize];
        for (int i=0; i<batch.length; i++) {
            batch[i] = Prng.nextInt(numExamples);
        }
        return batch;
    }
    
    /** Samples a batch of indices in the range [0, numExamples) without replacement. */
    public int[] sampleBatchWithoutReplacement() {
        int[] batch = new int[batchSize];
        for (int i=0; i<batch.length; i++) {
            if (cur == indices.length) {
                cur = 0;
            }
            if (cur == 0) {
                IntArrays.shuffle(indices);
            }
            batch[i] = indices[cur++];
        }
        return batch;
    }
        
}
