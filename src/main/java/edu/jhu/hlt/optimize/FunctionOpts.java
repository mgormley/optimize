package edu.jhu.hlt.optimize;

import edu.jhu.hlt.util.math.Vectors;

/**
 * Function operations.
 * 
 * @author mgormley
 */
public class FunctionOpts {

    public static DifferentiableFunction negate(DifferentiableFunction f) {
        return new NegateFunction(f);
    }
    
    public static DifferentiableFunction scale(DifferentiableFunction f, double multiplier) {
        return new ScaleFunction(f, multiplier);
    }

    public static Function negate(Function f) {
        return new NegateFunction1(f);
    }
    
    public static Function scale(Function f, double multiplier) {
        return new ScaleFunction1(f, multiplier);
    }
    
	public static class FunctionWithConstraints implements ConstrainedFunction {
		private Function f;
		private Bounds b;
		
		public FunctionWithConstraints(Function f, Bounds b) {
			this.f = f;
			this.b = b;
		}
		
		@Override
		public void setPoint(double[] point) {
			f.setPoint(point);
		}
		@Override
		public double[] getPoint() {
			return f.getPoint();
		}
		@Override
		public double getValue(double[] point) {
			return f.getValue(point);
		}
		@Override
		public double getValue() {
			return f.getValue();
		}
		@Override
		public int getNumDimensions() {
			return f.getNumDimensions();
		}
		@Override
		public Bounds getBounds() {
			return b;
		}
		@Override
		public void setBounds(Bounds b) {
			this.b = b;
		}
	}
	
	public static class DifferentiableFunctionWithConstraints implements ConstrainedDifferentiableFunction {
		
		private DifferentiableFunction f;
		private Bounds b;
		
		public DifferentiableFunctionWithConstraints(DifferentiableFunction f, Bounds b) {
			this.f = f;
			this.b = b;
		}
		
		@Override
		public void getGradient(double[] gradient) {
			f.getGradient(gradient);
		}
		@Override
		public void setPoint(double[] point) {
			f.setPoint(point);
		}
		@Override
		public double[] getPoint() {
			return f.getPoint();
		}
		@Override
		public double getValue(double[] point) {
			return f.getValue(point);
		}
		@Override
		public double getValue() {
			return f.getValue();
		}
		@Override
		public int getNumDimensions() {
			return f.getNumDimensions();
		}
		@Override
		public Bounds getBounds() {
			return b;
		}
		@Override
		public void setBounds(Bounds b) {
			this.b = b;
		}
		
	}
	
    /** Wrapper which negates the input function. */
    // TODO: Rename to NegateDifferentiableFunction.
    // TODO: Make private.
    public static class NegateFunction extends ScaleFunction implements DifferentiableFunction {
    
        public NegateFunction(DifferentiableFunction function) {
            super(function, -1.0);
        }
        
    }
    
    /** Wrapper which scales the input function. */
    // TODO: Rename to ScaleDifferentiableFunction.
    // TODO: Make private.
    public static class ScaleFunction implements DifferentiableFunction {
    
        private DifferentiableFunction function;
        private double multiplier;
        
        public ScaleFunction(DifferentiableFunction function, double multiplier) {
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

		@Override
		public double[] getPoint() {
			return function.getPoint();
		}

		@Override
		public double getValue(double[] point) {
			return multiplier*function.getValue(point);
		}
    
    }
    

    /** Wrapper which negates the input function. */
    // TODO: Drop the 1 from this name.
    private static class NegateFunction1 extends ScaleFunction1 implements Function {
    
        public NegateFunction1(Function function) {
            super(function, -1.0);
        }
        
    }
    
    /** Wrapper which scales the input function. */
    // TODO: Drop the 1 from this name.
    private static class ScaleFunction1 implements Function {
    
        private Function function;
        private double multiplier;
        
        public ScaleFunction1(Function function, double multiplier) {
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
        public int getNumDimensions() {
            return function.getNumDimensions();
        }

        @Override
        public double[] getPoint() {
            return function.getPoint();
        }

        @Override
        public double getValue(double[] point) {
            return multiplier*function.getValue(point);
        }
    
    }
    
    /** Wrapper which adds the input functions. */
    public static class AddFunctions implements DifferentiableFunction {
    
        private DifferentiableFunction[] functions;
        
        public AddFunctions(DifferentiableFunction... functions) {
            int numDims = functions[0].getNumDimensions();
            for (DifferentiableFunction f : functions) {
                if (numDims != f.getNumDimensions()) {
                    throw new IllegalArgumentException("Functions have different dimension.");
                }
            }
            this.functions = functions;
        }
        
        @Override
        public void setPoint(double[] point) {
            for (DifferentiableFunction function : functions) {
                function.setPoint(point);
            }
        }
        
        @Override
        public double getValue() {
            double sum = 0.0;
            for (DifferentiableFunction f : functions) {
                sum += f.getValue();                
            }
            return sum;
        }
    
        @Override
        public void getGradient(double[] gradient) {
            double[] g = new double[getNumDimensions()];
            for (DifferentiableFunction f : functions) {
                f.getGradient(g);
                Vectors.add(gradient, g);
            }
        }
    
        @Override
        public int getNumDimensions() {
            return functions[0].getNumDimensions();
        }

		@Override
		public double[] getPoint() {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public double getValue(double[] point) {
			// TODO Auto-generated method stub
			return 0;
		}
    
    }

}
