package edu.jhu.hlt.optimize;

import edu.jhu.hlt.util.math.Vectors;

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
        public void setPoint(double[] point) {
            function.setPoint(point);
        }
        
        @Override
        public double getValue(int[] batch) {
            return multiplier * function.getValue(batch);
        }
    
        @Override
        public void getGradient(int[] batch, double[] gradient) {
            function.getGradient(batch, gradient);
            Vectors.scale(gradient, multiplier);
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
		public double[] getPoint() {
			// TODO Auto-generated method stub
			return null;
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
        public void setPoint(double[] point) {
            for (DifferentiableBatchFunction function : functions) {
                function.setPoint(point);
            }
        }
        
        @Override
        public double getValue(int[] batch) {
            double sum = 0.0;
            for (DifferentiableBatchFunction f : functions) {
                sum += f.getValue(batch);                
            }
            return sum;
        }
    
        @Override
        public void getGradient(int[] batch, double[] gradient) {
            double[] g = new double[getNumDimensions()];
            for (DifferentiableBatchFunction f : functions) {
                f.getGradient(batch, g);
                Vectors.add(gradient, g);
            }
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
		public double[] getPoint() {
			// TODO Auto-generated method stub
			return null;
		}
    
    }

}
