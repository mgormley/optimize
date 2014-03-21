package edu.jhu.hlt.optimize.function;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;

import org.junit.Test;

import edu.jhu.util.Prng;
import edu.jhu.prim.util.JUnitUtils;
import edu.jhu.prim.vector.IntDoubleDenseVector;
import edu.jhu.prim.vector.IntDoubleVector;

public class SampleFunctionTest {

    private static class MockFunction extends AbstractDifferentiableBatchFunction {

        private int numExamples;
        private int numDims;
        public int[] selected;
        
        public MockFunction(int numExamples, int numDims) {
            this.numExamples = numExamples;
            this.numDims = numDims;
        }


        @Override
        public double getValue(IntDoubleVector point, int[] batch) {            
            selected = batch;
            return 0;
        }

        @Override
        public IntDoubleVector getGradient(IntDoubleVector point, int[] batch) {
            selected = batch;
            return null;
        }
        
        @Override
        public ValueGradient getValueGradient(IntDoubleVector point, int[] batch) {
            return new ValueGradient(getValue(point, batch), getGradient(point, batch));
        }

        @Override
        public int getNumDimensions() {
            return numDims;
        }

        @Override
        public int getNumExamples() {
            return numExamples;
        }
        
    }
    
    @Test
    public void testSampleFunction() {
        Prng.seed(123456789);
        MockFunction f = new MockFunction(100, 1000000);
        SampleFunction sf = new SampleFunction(f, 10);
        assertEquals(f.getNumDimensions(), sf.getNumDimensions());
        assertEquals(10, sf.getNumExamples());
        
        IntDoubleVector point = new IntDoubleDenseVector();
        int[] batch = new int[]{ 1, 3, 5, 7, 9 };

        int[] expectedBatch = new int[]{17, 78, 30, 81, 23};

        f.selected = null;
        sf.getValue(point, batch);
        System.out.println(Arrays.toString(f.selected));
        JUnitUtils.assertArrayEquals(expectedBatch, f.selected);
        
        f.selected = null;
        sf.getGradient(point, batch);
        JUnitUtils.assertArrayEquals(expectedBatch, f.selected);
        
        f.selected = null;
        sf.getValueGradient(point, batch);
        JUnitUtils.assertArrayEquals(expectedBatch, f.selected);
    }
    
}
