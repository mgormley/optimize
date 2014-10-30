package edu.jhu.hlt.optimize.function;

import edu.jhu.hlt.optimize.Optimizer;
import edu.jhu.hlt.optimize.functions.L1;
import edu.jhu.hlt.optimize.functions.L2;
import edu.jhu.prim.vector.IntDoubleVector;

public class DifferentiableFunctionOpts {

    public static DifferentiableFunction negate(DifferentiableFunction f) {
        return new NegateFunction(f);
    }

    public static DifferentiableFunction scale(DifferentiableFunction f, double multiplier) {
        return new ScaleFunction(f, multiplier);
    }

    public static class DifferentiableFunctionWithConstraints implements ConstrainedDifferentiableFunction {
        
        private DifferentiableFunction f;
        private Bounds b;
        
        public DifferentiableFunctionWithConstraints(DifferentiableFunction f, Bounds b) {
            this.f = f;
            this.b = b;
        }
        
        @Override
        public IntDoubleVector getGradient(IntDoubleVector pt) {
            return f.getGradient(pt);
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

        @Override
        public ValueGradient getValueGradient(IntDoubleVector point) {
            return f.getValueGradient(point);
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
        public IntDoubleVector getGradient(IntDoubleVector pt) {
            IntDoubleVector g = function.getGradient(pt);
            g.scale(multiplier);
            return g;
        }
    
        @Override
        public int getNumDimensions() {
            return function.getNumDimensions();
        }
    
    	@Override
    	public double getValue(IntDoubleVector point) {
    		return multiplier*function.getValue(point);
    	}
    
    	@Override
    	public ValueGradient getValueGradient(IntDoubleVector point) {
            ValueGradient vg = function.getValueGradient(point);
            vg.getGradient().scale(multiplier);     
            return new ValueGradient(vg.getValue() * multiplier, vg.getGradient());
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
        public double getValue(IntDoubleVector val) {
        	double sum = 0.0;
        	for(DifferentiableFunction f : functions) {
        		sum += f.getValue(val);
        	}
        	return sum;
        }
        
        @Override
        public IntDoubleVector getGradient(IntDoubleVector pt) {
        	IntDoubleVector ret = functions[0].getGradient(pt);
        	for(int i=1; i<functions.length; i++){
        		ret.add(functions[i].getGradient(pt));
        	}
        	return ret;
        }
    
        @Override
        public int getNumDimensions() {
            return functions[0].getNumDimensions();
        }
    
    	@Override
    	public ValueGradient getValueGradient(IntDoubleVector point) {
    	    double sum = 0.0;
            IntDoubleVector ret = null;
            for(int i=0; i<functions.length; i++){
                ValueGradient vg = functions[i].getValueGradient(point);
                sum += vg.getValue();
                if (i==0) {
                    ret = vg.getGradient();
                } else {
                    ret.add(vg.getGradient());
                }
            }
            return new ValueGradient(sum, ret);
    	}
    
    }

    /** Converts a standard optimizer to one which the given amount of l1 or l2 regularization. */
    public static Optimizer<DifferentiableFunction> getRegularizedOptimizer(final Optimizer<DifferentiableFunction> opt,
            final double l1Lambda, final double l2Lambda) {
        if (l1Lambda == 0 && l2Lambda == 0) {
            return opt;
        }
        return new Optimizer<DifferentiableFunction>() {
            
            @Override
            public boolean minimize(DifferentiableFunction function, IntDoubleVector point) {
                return optimize(function, point, false);
            }
            
            @Override
            public boolean maximize(DifferentiableFunction function, IntDoubleVector point) {
                return optimize(function, point, true);
            }
            
            public boolean optimize(DifferentiableFunction objective, IntDoubleVector point, boolean maximize) {
                DifferentiableFunction fn = getRegularizedFn(objective, maximize, l1Lambda, l2Lambda);
                
                if (!maximize) {
                    return opt.minimize(fn, point);   
                } else {
                    return opt.maximize(fn, point);
                }
            }
        };
    }

    public static DifferentiableFunction getRegularizedFn(DifferentiableFunction objective, boolean maximize,
            final double l1Lambda, final double l2Lambda) {
        L1 l1 = new L1(l1Lambda);
        L2 l2 = new L2(1.0 / l2Lambda);
        l1.setNumDimensions(objective.getNumDimensions());
        l2.setNumDimensions(objective.getNumDimensions());
        DifferentiableFunction br;
        if (l1Lambda != 0 && l2Lambda != 0) {
            br = new AddFunctions(l1, l2);
        } else if (l1Lambda != 0) {
            br = l1;
        } else if (l2Lambda != 0) {
            br = l2;
        } else {
            return objective;
        }

        DifferentiableFunction nbr = !maximize ? new NegateFunction(br) : br;
        DifferentiableFunction fn = new AddFunctions(objective, nbr);
        return fn;
    }
    
}
