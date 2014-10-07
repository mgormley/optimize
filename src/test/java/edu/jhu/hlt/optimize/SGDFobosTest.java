package edu.jhu.hlt.optimize;

import org.junit.Test;

import edu.jhu.hlt.optimize.BottouSchedule.BottouSchedulePrm;
import edu.jhu.hlt.optimize.SGD.SGDPrm;
import edu.jhu.hlt.optimize.SGDFobos.SGDFobosPrm;
import edu.jhu.hlt.optimize.function.DifferentiableBatchFunction;
import edu.jhu.hlt.optimize.functions.SumSquares;
import edu.jhu.hlt.util.JUnitUtils;
import edu.jhu.hlt.util.math.Vectors;
import edu.jhu.prim.vector.IntDoubleDenseVector;

public class SGDFobosTest extends AbstractBatchOptimizerTest {

    @Override
    protected Optimizer<DifferentiableBatchFunction> getOptimizer() {
        SGDFobosPrm prm = getOptimizerPrm();
        return new SGDFobos(prm);
    }

    protected SGDFobosPrm getOptimizerPrm() {
        SGDFobosPrm prm = new SGDFobosPrm();
        prm.sched.setEta0(0.1 * 10);
        prm.numPasses = 100;
        prm.batchSize = 1;
        prm.autoSelectLr = false;
        prm.l1Lambda = 0.0;
        return prm;
    }    

    @Test
    public void testL1RegularizedOffsetNegSumSquaresMax() {
        SGDFobosPrm prm = getOptimizerPrm();
        prm.l1Lambda = 1.0;
        Optimizer<DifferentiableBatchFunction> opt = new SGDFobos(prm);
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 0.4, -5, 11};
        opt.maximize(negate(bf(new SumSquares(offsets))), new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(new double[]{-0.0, 4.5, -10.5}, max, 1e-10);
    }

    @Test
    public void testL2RegularizedOffsetNegSumSquaresMax() {
        SGDFobosPrm prm = getOptimizerPrm();
        prm.l2Lambda = 1.0;
        Optimizer<DifferentiableBatchFunction> opt = new SGDFobos(prm);
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 0.4, -5, 11};
        opt.maximize(negate(bf(new SumSquares(offsets))), new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(new double[]{-0.266, 3.333, -7.333}, max, 1e-3);
    }
    

    @Test
    public void testL1RegularizedOffsetNegSumSquaresMin() {
        SGDFobosPrm prm = getOptimizerPrm();
        prm.l1Lambda = 1.0;
        Optimizer<DifferentiableBatchFunction> opt = new SGDFobos(prm);
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 0.4, -5, 11};
        opt.minimize(bf(new SumSquares(offsets)), new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(new double[]{-0.0, 4.5, -10.5}, max, 1e-10);
    }

    @Test
    public void testL2RegularizedOffsetNegSumSquaresMin() {
        SGDFobosPrm prm = getOptimizerPrm();
        prm.l2Lambda = 1.0;
        Optimizer<DifferentiableBatchFunction> opt = new SGDFobos(prm);
        double[] initial = new double[] { 9, 2, -7};
        double[] offsets = new double[] { 0.4, -5, 11};
        opt.minimize(bf(new SumSquares(offsets)), new IntDoubleDenseVector(initial));
        double[] max = initial;
        Vectors.scale(offsets, -1.0);
        JUnitUtils.assertArrayEquals(new double[]{-0.266, 3.333, -7.333}, max, 1e-3);
    }
    
}