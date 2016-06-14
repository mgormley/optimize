package edu.jhu.hlt.optimize;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.hlt.optimize.LBFGS_port.LBFGSCallback;
import edu.jhu.hlt.optimize.LBFGS_port.LBFGSPrm;
import edu.jhu.hlt.optimize.LBFGS_port.StatusCode;
import edu.jhu.hlt.optimize.function.DifferentiableFunction;
import edu.jhu.hlt.optimize.function.DifferentiableFunctionOpts;
import edu.jhu.hlt.optimize.function.ValueGradient;
import edu.jhu.prim.Primitives.MutableDouble;
import edu.jhu.prim.vector.IntDoubleDenseVector;
import edu.jhu.prim.vector.IntDoubleVector;

public class LBFGS implements Optimizer<DifferentiableFunction> {

    private static final long serialVersionUID = 1L;
    private static final Logger log = LoggerFactory.getLogger(LBFGS.class);

    private LBFGSPrm param;
    
    public LBFGS() { this(new LBFGSPrm()); }

    public LBFGS(LBFGSPrm param) { this.param = param; }

    @Override
    public boolean minimize(final DifferentiableFunction fn, IntDoubleVector xVec) {
        // The initial point.
        double[] xArr = new double[fn.getNumDimensions()];
        setArrayFromVector(xArr, xVec);

        MutableDouble fx = new MutableDouble(0);
        
        log.info(String.format("%8s %8s %8s %8s %8s %8s", "k", "fx", "xnorm", "gnorm", "step", "ls"));
        log.info(String.format("%8s-%8s-%8s-%8s-%8s-%8s", "------", "------", "------", "------", "------", "------"));
        LBFGSCallback cd = new LBFGSCallback() {
            
            @Override
            public double proc_evaluate(double[] x, double[] g, double step) {
                ValueGradient vg = fn.getValueGradient(new IntDoubleDenseVector(x));
                setArrayFromVector(g, vg.getGradient());
                return vg.getValue();
            }
            
            @Override
            public StatusCode proc_progress(double[] x, double[] g, double fx, double xnorm, double gnorm, double step,
                    int k, int ls) {
                log.info(String.format("%8d %8.2g %8.2g %8.2g %8.2g %8d", k, fx, xnorm, gnorm, step, ls));
                return StatusCode.LBFGS_CONTINUE;
            }
            
        };

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
