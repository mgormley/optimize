package edu.jhu.hlt.optimize;

import org.junit.Test;

import edu.jhu.hlt.optimize.SGD.SGDPrm;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.functions.SumSquares;
import edu.jhu.hlt.util.JUnitUtils;
import edu.jhu.hlt.util.math.Vectors;
import edu.jhu.prim.vector.IntDoubleDenseVector;

public class SGDTest extends AbstractBatchOptimizerTest {

    @Override
    protected Optimizer<DifferentiableBatchFunction> getOptimizer() {
        SGDPrm prm = new SGDPrm();
        prm.initialLr = 0.1 * 10;
        prm.numPasses = 100;
        prm.batchSize = 1;
        prm.autoSelectLr = false;
        return new SGD(prm);
    }
    
    @Test
    public void testSgdAutoSelectLr() {
        {
            // Test with the initial learning rate too small
            SGDPrm prm = new SGDPrm();
            prm.initialLr = 0.005;
            prm.lambda = 0.01;
            prm.lambda = 0.1;
            prm.numPasses = 7;
            prm.batchSize = 1;
            prm.autoSelectLr = true;
            SGD opt = new SGD(prm);
            
            double[] initial = new double[] { 9, 2, -7};
            double[] offsets = new double[] { 3, -5, 11};
            opt.maximize(negate(bf(new SumSquares(offsets))), new IntDoubleDenseVector(initial));
            double[] max = initial;
            Vectors.scale(offsets, -1.0);
            JUnitUtils.assertArrayEquals(offsets, max, 1e-1);        
        }

        {
            // Test with the initial learning rate too large
            SGDPrm prm = new SGDPrm();
            prm.initialLr = 10;
            prm.lambda = 0.01;
            prm.numPasses = 7;
            prm.batchSize = 1;
            prm.autoSelectLr = true;
            SGD opt = new SGD(prm);
            
            double[] initial = new double[] { 9, 2, -7};
            double[] offsets = new double[] { 3, -5, 11};
            opt.maximize(negate(bf(new SumSquares(offsets))), new IntDoubleDenseVector(initial));
            double[] max = initial;
            Vectors.scale(offsets, -1.0);
            JUnitUtils.assertArrayEquals(offsets, max, 1e-1);        
        }
    }

}