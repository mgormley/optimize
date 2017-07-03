package edu.jhu.hlt.optimize.function;

import edu.jhu.hlt.optimize.Optimizer;
import edu.jhu.hlt.optimize.functions.L1;
import edu.jhu.hlt.optimize.functions.L2;
import edu.jhu.prim.vector.IntDoubleDenseVector;
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
    public static class ScaleFunction extends AbstractDifferentiableBatchFunction implements
            DifferentiableBatchFunction, NonstationaryFunction {
    
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
            ValueGradient vg = function.getValueGradient(point, batch);
            vg.getGradient().scale(multiplier);     
            return new ValueGradient(vg.getValue() * multiplier, vg.getGradient());
        }

        @Override
        public void updatateIterAndMax(int curIter, int maxIter) {
            if (function instanceof NonstationaryFunction) {
                ((NonstationaryFunction) function).updatateIterAndMax(curIter, maxIter);
            }
        }
    
    }
    
    /** Wrapper which adds the input functions. */
    public static class AddFunctions extends AbstractDifferentiableBatchFunction implements
            DifferentiableBatchFunction, NonstationaryFunction {
    
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
            // When adding up the gradients find one that is an IntDoubleDenseVector 
            // if possible and add into that.
            IntDoubleVector[] grads = new IntDoubleVector[functions.length];
            int retIdx = 0;
            for (int i=0; i<functions.length; i++) {
                grads[i] = functions[i].getGradient(point, batch);
                if (grads[i] instanceof IntDoubleDenseVector) {
                    retIdx = i;
                }
            }
            IntDoubleVector ret = grads[retIdx];
            for(int i=0; i<functions.length; i++){
                if (i != retIdx) {
                    ret.add(grads[i]);
                }
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
            double sum = 0.0;
            IntDoubleVector[] grads = new IntDoubleVector[functions.length];
            int retIdx = 0;
            for (int i=0; i<functions.length; i++) {
                ValueGradient vg = functions[i].getValueGradient(point, batch);
                sum += vg.getValue();
                grads[i] = vg.getGradient();
                if (grads[i] instanceof IntDoubleDenseVector) {
                    retIdx = i;
                }
            }
            IntDoubleVector ret = grads[retIdx];
            for(int i=0; i<functions.length; i++){
                if (i != retIdx) {
                    ret.add(grads[i]);
                }
            }
            return new ValueGradient(sum, ret);
        }

        @Override
        public void updatateIterAndMax(int curIter, int maxIter) {
            for (DifferentiableBatchFunction f : functions) {
                if (f instanceof NonstationaryFunction) {
                    ((NonstationaryFunction) f).updatateIterAndMax(curIter, maxIter);
                }
            }
        }
    
    }

    public static Optimizer<DifferentiableBatchFunction> getRegularizedOptimizer(
            final Optimizer<DifferentiableBatchFunction> opt, final double l1Lambda, final double l2Lambda) {
        if (l1Lambda == 0 && l2Lambda == 0) {
            return opt;
        }
        return new Optimizer<DifferentiableBatchFunction>() {

            @Override
            public boolean minimize(DifferentiableBatchFunction objective, IntDoubleVector point) {
                DifferentiableBatchFunction fn = getRegularizedFn(objective, false, l1Lambda, l2Lambda);                
                return opt.minimize(fn, point);   
            }

        };
    }

    public static DifferentiableBatchFunction getRegularizedFn(DifferentiableBatchFunction objective,
            boolean maximize, final double l1Lambda, final double l2Lambda) {
        L1 l1 = new L1(l1Lambda);
        L2 l2 = new L2(l2Lambda);
        l1.setNumDimensions(objective.getNumDimensions());
        l2.setNumDimensions(objective.getNumDimensions());
        DifferentiableFunction reg;
        if (l1Lambda != 0 && l2Lambda != 0) {
            reg = new DifferentiableFunctionOpts.AddFunctions(l1, l2);
        } else if (l1Lambda != 0) {
            reg = l1;
        } else if (l2Lambda != 0) {
            reg = l2;
        } else {
            return objective;
        }
        
        DifferentiableBatchFunction br = new FunctionAsBatchFunction(reg, objective.getNumExamples());
        DifferentiableBatchFunction nbr = !maximize ? new NegateFunction(br) : br;
        DifferentiableBatchFunction fn = new AddFunctions(objective, nbr);
        return fn;
    }

}
