package edu.jhu.optimize;

import edu.jhu.util.math.Vectors;

public class FunctionOpts {

    /** Wrapper which negates the input function. */
    public static class NegateFunction extends ScaleFunction implements Function {
    
        public NegateFunction(Function function) {
            super(function, -1.0);
        }
        
    }
    
    /** Wrapper which scales the input function. */
    public static class ScaleFunction implements Function {
    
        private Function function;
        private double multiplier;
        
        public ScaleFunction(Function function, double multiplier) {
            this.function = function;
            this.multiplier = multiplier;
        }
        
        @Override
        public void setPoint(double[] point) {
            function.setPoint(point);
        }
        
        @Override
        public double getValue() {
            return multiplier * function.getValue();
        }
    
        @Override
        public void getGradient(double[] gradient) {
            function.getGradient(gradient);
            Vectors.scale(gradient, multiplier);
        }
    
        @Override
        public int getNumDimensions() {
            return function.getNumDimensions();
        }
    
    }
    
    /** Wrapper which adds the input functions. */
    public static class AddFunctions implements Function {
    
        private Function[] functions;
        
        public AddFunctions(Function... functions) {
            int numDims = functions[0].getNumDimensions();
            for (Function f : functions) {
                if (numDims != f.getNumDimensions()) {
                    throw new IllegalArgumentException("Functions have different dimension.");
                }
            }
            this.functions = functions;
        }
        
        @Override
        public void setPoint(double[] point) {
            for (Function function : functions) {
                function.setPoint(point);
            }
        }
        
        @Override
        public double getValue() {
            double sum = 0.0;
            for (Function f : functions) {
                sum += f.getValue();                
            }
            return sum;
        }
    
        @Override
        public void getGradient(double[] gradient) {
            double[] g = new double[getNumDimensions()];
            for (Function f : functions) {
                f.getGradient(g);
                Vectors.add(gradient, g);
            }
        }
    
        @Override
        public int getNumDimensions() {
            return functions[0].getNumDimensions();
        }
    
    }

}
