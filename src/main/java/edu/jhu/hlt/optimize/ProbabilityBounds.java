package edu.jhu.hlt.optimize;

/**
 * 
 * 
 * @author fmof
 */
public class ProbabilityBounds extends Bounds {
    public ProbabilityBounds(int dim) {
	super(dim);
	for(int i=0;i<dim;i++){
	    super.A[i]=0.0;
	    super.B[i]=1.0;
	}
    }

    @Override
    public double transformFromUnitInterval(int i, double d){
	return d;
    }
}
