package edu.jhu.hlt.optimize;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.hlt.optimize.LBFGS_port.StatusCode;
import edu.jhu.hlt.optimize.LBFGS_port.callback_data_t;
import edu.jhu.hlt.optimize.LBFGS_port.lbfgs_parameter_t;
import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.DifferentiableFunctionOpts;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.prim.Primitives.MutableDouble;
import edu.jhu.prim.arrays.DoubleArrays;
import edu.jhu.prim.vector.IntDoubleDenseVector;
import edu.jhu.prim.vector.IntDoubleVector;

public class LBFGS implements Optimizer<DifferentiableFunction> {

    private static final long serialVersionUID = 1L;
    private static final Logger log = LoggerFactory.getLogger(LBFGS.class);

    private lbfgs_parameter_t param;
    
    public LBFGS() { this(new lbfgs_parameter_t()); }

    public LBFGS(lbfgs_parameter_t param) { this.param = param; }

    @Override
    public boolean maximize(DifferentiableFunction fn, IntDoubleVector x) {
        return minimize(DifferentiableFunctionOpts.negate(fn), x);
    }

    @Override
    public boolean minimize(final DifferentiableFunction fn, IntDoubleVector xVec) {
        // The initial point.
        double[] xArr = new double[fn.getNumDimensions()];
        setArrayFromVector(xArr, xVec);

        MutableDouble fx = new MutableDouble(0);
        
        log.info(String.format("%8s %8s %8s %8s %8s %8s", "k", "fx", "xnorm", "gnorm", "step", "ls"));
        log.info(String.format("%8s-%8s-%8s-%8s-%8s-%8s", "------", "------", "------", "------", "------", "------"));
        callback_data_t cd = new callback_data_t() {
            
            @Override
            double proc_evaluate(Object instance, double[] x, double[] g, int n, double step) {
                ValueGradient vg = fn.getValueGradient(new IntDoubleDenseVector(x));
                setArrayFromVector(g, vg.getGradient());
                return vg.getValue();
            }
            
            @Override
            StatusCode proc_progress(Object instance, double[] x, double[] g, double fx, double xnorm, double gnorm, double step,
                    int n, int k, int ls) {
                log.info(String.format("%8d %8.2g %8.2g %8.2g %8.2g %8d", k, fx, xnorm, gnorm, step, ls));
                return StatusCode.LBFGS_CONTINUE;
            }
            
        };
        cd.n = xArr.length;

        // Minimize.
        StatusCode ret = LBFGS_port.lbfgs(xArr, fx, cd, param);
        if (ret.ret != 0) {
            log.warn("Error from LBFGS: " + ret);
        }
        
        // Return the minimum.
        setVectorFromArray(xVec, xArr);
        return true;
    }

    private void setArrayFromVector(double[] point, IntDoubleVector x) {
        for (int i=0; i<point.length; i++) {
            point[i] = x.get(i); 
        }
    }

    private void setVectorFromArray(IntDoubleVector x, double[] point) {
        for (int i=0; i<point.length; i++) {
            x.set(i, point[i]); 
        }
    }

}
