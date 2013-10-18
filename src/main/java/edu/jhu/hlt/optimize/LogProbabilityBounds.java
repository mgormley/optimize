package edu.jhu.hlt.optimize;

/**
 * 
 * 
 * @author fmof
 */
public class LogProbabilityBounds extends Bounds {
    public LogProbabilityBounds(int dim) {
	super(dim);
	for(int i=0;i<dim;i++){
	    super.A[i]=Double.NEGATIVE_INFINITY;
	    super.B[i]=0.0;
	}
    }

    @Override
    public double transformFromUnitInterval(int i, double d){
	return Math.log(d);
    }
}
