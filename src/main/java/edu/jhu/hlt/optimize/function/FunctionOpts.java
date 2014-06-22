package edu.jhu.hlt.optimize.function;

import edu.jhu.hlt.util.math.Vectors;
import edu.jhu.prim.vector.IntDoubleVector;

/**
 * Function operations.
 * 
 * @author mgormley
 */
public class FunctionOpts {

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
		public double getValue(IntDoubleVector point) {
			return f.getValue(point);
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
        public int getNumDimensions() {
            return function.getNumDimensions();
        }


        @Override
        public double getValue(IntDoubleVector pt) {
            return multiplier*function.getValue(pt);
        }
    
    }

}
