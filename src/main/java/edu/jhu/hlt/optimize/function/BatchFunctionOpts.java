package edu.jhu.hlt.optimize.function;

import edu.jhu.hlt.util.math.Vectors;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * Batch function operations. 
 * 
 * @author mgormley
 */
public class BatchFunctionOpts {

    /** Wrapper which negates the input function. */
    public static class NegateFunction extends ScaleFunction implements DifferentiableBatchFunction {
        
        public NegateFunction(DifferentiableBatchFunction function) {
            super(function, -1.0);
        }
    
    }

    /** Wrapper which scales the input function. */
    public static class ScaleFunction extends AbstractDifferentiableBatchFunction implements DifferentiableBatchFunction {
    
        private DifferentiableBatchFunction function;
        private double multiplier;
        
        public ScaleFunction(DifferentiableBatchFunction function, double multiplier) {
            this.function = function;
            this.multiplier = multiplier;
        }
        
        @Override
        public double getValue(IntDoubleVector point, int[] batch) {
            return multiplier * function.getValue(point, batch);
        }
    
        @Override
        public IntDoubleVector getGradient(IntDoubleVector point, int[] batch) {
            IntDoubleVector gradient = function.getGradient(point, batch);
            gradient.scale(multiplier);
            return gradient;
        }
    
        @Override
        public int getNumDimensions() {
            return function.getNumDimensions();
        }

        @Override
        public int getNumExamples() {
            return function.getNumExamples();
        }

        @Override
        public ValueGradient getValueGradient(IntDoubleVector point, int[] batch) {
            return new ValueGradient(getValue(point, batch), getGradient(point, batch));
        }
    
    }
    
    /** Wrapper which adds the input functions. */
    public static class AddFunctions extends AbstractDifferentiableBatchFunction implements DifferentiableBatchFunction {
    
        private DifferentiableBatchFunction[] functions;
        
        public AddFunctions(DifferentiableBatchFunction... functions) {
            int numDims = functions[0].getNumDimensions();
            int numExs = functions[0].getNumExamples();
            for (DifferentiableBatchFunction f : functions) {
                if (numDims != f.getNumDimensions()) {
                    throw new IllegalArgumentException("Functions have different dimension.");
                }
                if (numExs != f.getNumExamples()) {
                    throw new IllegalArgumentException("Functions have different numbers of examples.");
                }
            }
            this.functions = functions;
        }
        
        @Override
        public double getValue(IntDoubleVector point, int[] batch) {
            double sum = 0.0;
            for (DifferentiableBatchFunction f : functions) {
                sum += f.getValue(point, batch);                
            }
            return sum;
        }
    
        @Override
        public IntDoubleVector getGradient(IntDoubleVector point, int[] batch) {
            IntDoubleVector ret = functions[0].getGradient(point, batch);
            for(int i=1; i<functions.length; i++){
                ret.add(functions[i].getGradient(point, batch));
            }
            return ret;
        }
    
        @Override
        public int getNumDimensions() {
            return functions[0].getNumDimensions();
        }

        @Override
        public int getNumExamples() {
            return functions[0].getNumExamples();
        }

        @Override
        public ValueGradient getValueGradient(IntDoubleVector point, int[] batch) {
            return new ValueGradient(getValue(point, batch), getGradient(point, batch));
        }
    
    }

}
